"""
LLM Evaluator for Invoice Data Extraction 

This module evaluates extracted invoice JSON data against curated ground truth CSV
using GroundX LLM for semantic comparison of field values.

"""

import json
import pandas as pd
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import hashlib

# GroundX imports
from groundx.extract.settings.settings import AgentSettings
from groundx.extract.services.logger import Logger
from groundx.extract.agents import AgentCode


class Judgment(str, Enum):
    """Enumeration for field judgment results."""
    RIGHT = "RIGHT"
    WRONG = "WRONG"
    SKIPPED = "SKIPPED"


@dataclass
class FieldResult:
    """Data class representing the evaluation result for a single field."""
    field_name: str
    extracted_value: str
    ground_truth_value: str
    judgment: Judgment
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "field_name": self.field_name,
            "extracted_value": self.extracted_value,
            "ground_truth_value": self.ground_truth_value,
            "judgment": self.judgment.value,
            "notes": self.notes
        }


@dataclass
class EvaluationReport:
    """Data class representing a complete evaluation report."""
    evaluation_id: str
    json_file: str
    statement_id: str
    evaluation_timestamp: str
    processing_time_seconds: float
    total_fields_evaluated: int
    fields_right: int
    fields_wrong: int
    fields_skipped: int
    accuracy_percentage: float
    field_results: List[FieldResult]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "evaluation_id": self.evaluation_id,
            "json_file": self.json_file,
            "statement_id": self.statement_id,
            "evaluation_timestamp": self.evaluation_timestamp,
            "processing_time_seconds": self.processing_time_seconds,
            "summary": {
                "total_fields_evaluated": self.total_fields_evaluated,
                "fields_right": self.fields_right,
                "fields_wrong": self.fields_wrong,
                "fields_skipped": self.fields_skipped,
                "accuracy_percentage": self.accuracy_percentage
            },
            "field_details": {
                result.field_name: result.to_dict() 
                for result in self.field_results
            },
            "metadata": self.metadata
        }


class LLMEvaluator:
    """
    Evaluates extracted invoice JSON data against curated ground truth CSV using LLM judgment.
    
    The AgentCode is initialized once with a system prompt containing evaluation rules
    and examples. Each field evaluation uses the same agent instance for efficiency.
    
    Attributes:
        ground_truth_df: DataFrame containing ground truth data from CSV
        agent_settings: GroundX AgentSettings for LLM configuration
        agent: AgentCode instance for LLM calls (initialized once)
        logger: GroundX Logger instance
        cache: Dictionary for caching LLM judgments to avoid redundant calls
    
    Example:
        >>> evaluator = LLMEvaluator(csv_path="data/curated.statements.csv")
        >>> report = evaluator.evaluate("data/invoice-extract.json")
        >>> print(f"Accuracy: {report.accuracy_percentage}%")
    """
    
    # System prompt for batched evaluation
    SYSTEM_PROMPT = """You are an expert evaluator for invoice data extraction accuracy.

Your task is to compare multiple EXTRACTED VALUES against their GROUND TRUTH VALUES and determine if each pair is semantically equivalent.

## EVALUATION RULES

### 1. FORMATTING DIFFERENCES (Acceptable - Mark as RIGHT)
- Spacing variations: "John Smith" = "John  Smith"
- Case differences: "ACME Corp" = "acme corp" = "Acme Corp"
- Punctuation: "123-456-7890" = "123.456.7890" = "(123) 456-7890"

### 2. NUMERICAL EQUIVALENCES (Acceptable - Mark as RIGHT)
- Currency formatting: "1000" = "1,000" = "1000.00" = "$1,000.00" = "USD 1000"
- Percentages: "5%" = "5.0%" = "0.05" (when context is percentage)
- Leading zeros: "007" = "7"

### 3. DATE FORMAT VARIATIONS (Acceptable - Mark as RIGHT)
- "2024-01-15" = "01/15/2024" = "15/01/2024" = "January 15, 2024" = "Jan 15, 2024" = "15-Jan-2024"

### 4. COMMON OCR ERRORS (Consider - Usually RIGHT if meaning preserved)
- "O" vs "0" (letter O vs zero)
- "I" vs "1" vs "l" (letter I vs one vs lowercase L)
- "S" vs "5"
- "B" vs "8"

### 5. MUST BE WRONG
- Completely different values
- Missing significant information
- Different dates (not just format)
- Different amounts
- Wrong account/invoice numbers

## EXAMPLES

Example 1:
FIELD: account_number
EXTRACTED: "ACC-12345"
GROUND TRUTH: "ACC12345"
JUDGMENT: RIGHT
(Same number, minor formatting difference)

Example 2:
FIELD: amount_due
EXTRACTED: "$1,234.56"
GROUND TRUTH: "1234.56"
JUDGMENT: RIGHT
(Same amount, currency symbol and comma are formatting)

Example 3:
FIELD: due_date
EXTRACTED: "2024-03-15"
GROUND TRUTH: "March 15, 2024"
JUDGMENT: RIGHT
(Same date, different format)

Example 4:
FIELD: invoice_number
EXTRACTED: "INV-001"
GROUND TRUTH: "INV-002"
JUDGMENT: WRONG
(Different invoice numbers)

Example 5:
FIELD: vendor_name
EXTRACTED: "ACME Corporation"
GROUND TRUTH: "Acme Corp."
JUDGMENT: RIGHT
(Same company, minor variations)

Example 6:
FIELD: total_amount
EXTRACTED: "500.00"
GROUND TRUTH: "5000.00"
JUDGMENT: WRONG
(Different amounts - order of magnitude difference)

Example 7:
FIELD: account_number
EXTRACTED: "l2345678"
GROUND TRUTH: "12345678"
JUDGMENT: RIGHT
(OCR error: lowercase L confused with 1)

## RESPONSE FORMAT

You MUST respond with a valid JSON object containing judgments for ALL fields.
Format:
```json
{
  "field_name_1": "RIGHT",
  "field_name_2": "WRONG",
  "field_name_3": "RIGHT"
}
```

CRITICAL RULES:
- Include EVERY field from the input
- Use ONLY "RIGHT" or "WRONG" as values
- Return valid JSON only, no explanations or additional text
- Field names in response must EXACTLY match the input field names
"""

    # Field name mappings for common variations (JSON key -> CSV column)
    FIELD_MAPPINGS: Dict[str, str] = {
        "accountNumber": "account_number",
        "amountDue": "amount_due",
        "dueDate": "due_date",
        "invoiceNumber": "invoice_number",
        "invoiceDate": "invoice_date",
        "totalAmount": "total_amount",
        "vendorName": "vendor_name",
        "customerName": "customer_name",
    }
    
    # Fields to exclude from comparison
    EXCLUDED_FIELDS: set = {"file_name", "statement_id", "extraction_timestamp", "source_url"}
    
    def __init__(
        self, 
        csv_path: str, 
        agent_settings: Optional[AgentSettings] = None,
        output_dir: str = "evaluation_reports",
        enable_caching: bool = True,
        enable_tracing: bool = False,
        log_level: str = "INFO"
    ):
        """Initialize the LLM evaluator with ground truth data and AgentCode."""
        self.logger = Logger(__name__, log_level)
        self.logger.info_msg(f"Initializing LLMEvaluator (Batched) with CSV: {csv_path}")
        
        self._validate_csv_path(csv_path)
        self.ground_truth_df = self._load_ground_truth(csv_path)
        
        self.agent_settings = agent_settings or AgentSettings()
        self.agent = self._create_agent()
        
        self.output_dir = output_dir
        self.enable_caching = enable_caching
        self._document_cache: Dict[str, Dict[str, Judgment]] = {}
        
        os.makedirs(self.output_dir, exist_ok=True)

        if enable_tracing:
            self._setup_tracing()
    
    def _create_agent(self) -> AgentCode:
        """Create and configure the AgentCode instance."""
        self.agent_settings.imports = []
        agent = AgentCode(
            settings=self.agent_settings,
            log=self.logger
        )
        return agent

    def _setup_tracing(self) -> None:
        """Setup OpenTelemetry tracing for LLM calls."""
        try:
            from openinference.instrumentation.smolagents import SmolagentsInstrumentor
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor
            
            trace.set_tracer_provider(TracerProvider())
            tracer_provider = trace.get_tracer_provider()
            span_processor = BatchSpanProcessor(ConsoleSpanExporter())
            tracer_provider.add_span_processor(span_processor)
            SmolagentsInstrumentor().instrument(tracer_provider=tracer_provider)
            self.logger.info_msg("OpenTelemetry tracing initialized")
        except ImportError as e:
            self.logger.error_msg(f"Failed to import tracing packages: {e}")
            raise
    
    def _validate_csv_path(self, csv_path: str) -> None:
        """Validate that the CSV path exists and is readable."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        if not os.path.isfile(csv_path):
            raise ValueError(f"Path is not a file: {csv_path}")
    
    def _load_ground_truth(self, csv_path: str) -> pd.DataFrame:
        """Load and validate ground truth CSV data."""
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.lower()
            
            if df.empty:
                raise ValueError("CSV file is empty")
            if 'statement_id' not in df.columns:
                raise ValueError("CSV missing required 'statement_id' column")
            
            df['statement_id'] = df['statement_id'].astype(str)
            return df
        except pd.errors.EmptyDataError:
            raise ValueError(f"CSV file is empty: {csv_path}")
    
    def _match_json_to_csv(self, json_data: Dict[str, Any]) -> pd.Series:
        """Match JSON file to the correct CSV row using file_name/statement_id."""
        file_name = json_data.get('file_name', '')
        if not file_name:
            raise ValueError("JSON data missing 'file_name' field")
        
        statement_id = Path(file_name).stem
        matching_rows = self.ground_truth_df[
            self.ground_truth_df['statement_id'] == statement_id
        ]
        
        if len(matching_rows) == 0:
            matching_rows = self.ground_truth_df[
                self.ground_truth_df['statement_id'].str.lower() == statement_id.lower()
            ]
            
        if len(matching_rows) == 0:
            available_ids = self.ground_truth_df['statement_id'].head(5).tolist()
            raise ValueError(f"No CSV row found for statement_id: '{statement_id}'. Available: {available_ids}")
        
        return matching_rows.iloc[0]
    
    def _normalize_field_name(self, field_name: str) -> str:
        """Normalize a field name to match CSV column conventions."""
        if field_name in self.FIELD_MAPPINGS:
            return self.FIELD_MAPPINGS[field_name]
        return re.sub(r'(?<!^)(?=[A-Z])', '_', field_name).lower()
    
    def _clean_field_value(self, value: Any) -> str:
        """Clean and normalize field values for comparison."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        cleaned = str(value).strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.replace('\n', ' ').replace('\r', '')
    
    def _get_comparable_fields(
        self, 
        json_data: Dict[str, Any], 
        ground_truth: pd.Series
    ) -> List[Tuple[str, str, str, str]]:
        """Get list of fields that can be compared between JSON and CSV."""
        comparable_fields = []
        csv_columns = set(ground_truth.index) - {'statement_id'}
        json_keys = set(json_data.keys()) - self.EXCLUDED_FIELDS
        
        for json_key in json_keys:
            if json_key in csv_columns:
                csv_col = json_key
            else:
                normalized_key = self._normalize_field_name(json_key)
                if normalized_key in csv_columns:
                    csv_col = normalized_key
                else:
                    continue
            
            extracted_value = self._clean_field_value(json_data.get(json_key, ""))
            ground_truth_value = self._clean_field_value(ground_truth.get(csv_col, ""))
            comparable_fields.append((json_key, csv_col, extracted_value, ground_truth_value))
        
        return comparable_fields
    
    def _create_batched_prompt(
        self, 
        fields_to_evaluate: List[Tuple[str, str, str]]
    ) -> str:
        """
        Create a single prompt for evaluating all fields at once.
        
        Args:
            fields_to_evaluate: List of (field_name, extracted_value, ground_truth_value)
            
        Returns:
            Formatted prompt for batch evaluation
        """
        field_entries = []
        for field_name, extracted, ground_truth in fields_to_evaluate:
            entry = f"""Field: {field_name}
  EXTRACTED: "{extracted}"
  GROUND TRUTH: "{ground_truth}" """
            field_entries.append(entry)
        
        fields_block = "\n\n".join(field_entries)
        field_names = [f[0] for f in fields_to_evaluate]
        
        prompt = f"""Evaluate the following {len(fields_to_evaluate)} field extractions:

{fields_block}

Return a JSON object with judgments for these exact fields: {field_names}
Each value must be "RIGHT" or "WRONG".

JSON response:"""
        
        return prompt
    
    def _parse_batched_response(
        self, 
        response_text: str, 
        expected_fields: List[str]
    ) -> Dict[str, Judgment]:
        """
        Parse the LLM's batched JSON response into field judgments.
        
        Args:
            response_text: Raw LLM response
            expected_fields: List of field names we expect in the response
            
        Returns:
            Dictionary mapping field names to Judgment enums
        """
        # Try to extract JSON from response
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if not json_match:
            # Try to find JSON with nested braces
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        
        if not json_match:
            self.logger.error_msg(f"No JSON found in response: {response_text[:200]}")
            # Return all WRONG as fallback
            return {field: Judgment.WRONG for field in expected_fields}
        
        try:
            json_str = json_match.group()
            # Clean up common issues
            json_str = json_str.replace("'", '"')
            parsed = json.loads(json_str)
            
            judgments = {}
            for field_name in expected_fields:
                if field_name in parsed:
                    value = parsed[field_name].upper().strip()
                    if value == "RIGHT":
                        judgments[field_name] = Judgment.RIGHT
                    else:
                        judgments[field_name] = Judgment.WRONG
                else:
                    # Field missing from response - try case-insensitive match
                    found = False
                    for key in parsed:
                        if key.lower() == field_name.lower():
                            value = parsed[key].upper().strip()
                            judgments[field_name] = Judgment.RIGHT if value == "RIGHT" else Judgment.WRONG
                            found = True
                            break
                    if not found:
                        self.logger.warning_msg(f"Field '{field_name}' missing from LLM response")
                        judgments[field_name] = Judgment.WRONG
            
            return judgments
            
        except json.JSONDecodeError as e:
            self.logger.error_msg(f"JSON parse error: {e}, response: {response_text[:200]}")
            return {field: Judgment.WRONG for field in expected_fields}
    
    def _call_llm_batched(
        self, 
        fields_to_evaluate: List[Tuple[str, str, str]]
    ) -> Dict[str, Judgment]:
        """
        Make a single LLM call to evaluate all fields.
        
        Args:
            fields_to_evaluate: List of (field_name, extracted_value, ground_truth_value)
            
        Returns:
            Dictionary mapping field names to Judgment enums
        """
        if not fields_to_evaluate:
            return {}
        
        prompt = self._create_batched_prompt(fields_to_evaluate)
        expected_fields = [f[0] for f in fields_to_evaluate]
        
        self.logger.debug_msg(f"Calling LLM for batch evaluation of {len(fields_to_evaluate)} fields")
        
        try:
            response = self.agent.run(prompt)
            
            if isinstance(response, str):
                response_text = response
            elif hasattr(response, 'text'):
                response_text = response.text
            elif hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            self.logger.debug_msg(f"LLM response: {response_text[:500]}")
            
            return self._parse_batched_response(response_text, expected_fields)
            
        except Exception as e:
            self.logger.error_msg(f"LLM batch call failed: {str(e)}")
            return {field: Judgment.WRONG for field in expected_fields}
    
    def _compare_fields(
        self, 
        json_data: Dict[str, Any], 
        ground_truth: pd.Series
    ) -> List[FieldResult]:
        """
        Compare all fields between JSON and ground truth using a SINGLE LLM call.
        
        Args:
            json_data: Extracted JSON data
            ground_truth: Ground truth row from CSV
            
        Returns:
            List of FieldResult objects
        """
        results = []
        comparable_fields = self._get_comparable_fields(json_data, ground_truth)
        
        # Separate fields into those needing LLM evaluation and those that can be skipped/exact-matched
        fields_for_llm = []  # (field_name, extracted, ground_truth)
        exact_matches = []   # FieldResult objects
        skipped_fields = []  # FieldResult objects
        
        for json_key, csv_col, extracted_value, ground_truth_value in comparable_fields:
            field_name = csv_col
            
            # Skip if both empty
            if not extracted_value and not ground_truth_value:
                skipped_fields.append(FieldResult(
                    field_name=field_name,
                    extracted_value=extracted_value,
                    ground_truth_value=ground_truth_value,
                    judgment=Judgment.SKIPPED,
                    notes="Both values empty"
                ))
                continue
            
            # Quick exact match check (case-insensitive)
            if extracted_value.lower() == ground_truth_value.lower():
                exact_matches.append(FieldResult(
                    field_name=field_name,
                    extracted_value=extracted_value,
                    ground_truth_value=ground_truth_value,
                    judgment=Judgment.RIGHT,
                    notes="Exact match (case-insensitive)"
                ))
                continue
            
            # Needs LLM evaluation
            fields_for_llm.append((field_name, extracted_value, ground_truth_value))
        
        self.logger.debug_msg(
            f"Field breakdown: {len(exact_matches)} exact matches, "
            f"{len(skipped_fields)} skipped, {len(fields_for_llm)} need LLM"
        )
        
        # Make single LLM call for all fields needing evaluation
        llm_judgments = {}
        if fields_for_llm:
            llm_judgments = self._call_llm_batched(fields_for_llm)
        
        # Build results for LLM-evaluated fields
        llm_results = []
        for field_name, extracted_value, ground_truth_value in fields_for_llm:
            judgment = llm_judgments.get(field_name, Judgment.WRONG)
            llm_results.append(FieldResult(
                field_name=field_name,
                extracted_value=extracted_value,
                ground_truth_value=ground_truth_value,
                judgment=judgment,
                notes="LLM semantic comparison (batched)"
            ))
            
            if judgment == Judgment.WRONG:
                self.logger.info_msg(
                    f"MISMATCH: {field_name} - "
                    f"extracted='{extracted_value}' vs ground_truth='{ground_truth_value}'"
                )
        
        # Combine all results
        results = exact_matches + llm_results + skipped_fields
        return results
    
    def _load_json_data(self, json_path: str) -> Dict[str, Any]:
        """Load JSON data from local path or URL."""
        if json_path.startswith(('http://', 'https://')):
            import requests
            try:
                response = requests.get(json_path, timeout=30)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                raise ValueError(f"Failed to fetch URL: {json_path} - {str(e)}")
        else:
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"JSON file not found: {json_path}")
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    
    def evaluate(
        self, 
        json_path: str,
        save_report: bool = True
    ) -> EvaluationReport:
        """
        Evaluate a single JSON file against ground truth using ONE LLM call.
        
        Args:
            json_path: Path (local or URL) to the extracted JSON file
            save_report: Whether to save the report to a file
            
        Returns:
            EvaluationReport object containing all evaluation details
        """
        start_time = datetime.now()
        self.logger.info_msg(f"Starting evaluation for: {json_path}")
        
        json_data = self._load_json_data(json_path)
        ground_truth_row = self._match_json_to_csv(json_data)
        statement_id = Path(json_data.get('file_name', '')).stem
        
        # Single LLM call happens inside _compare_fields
        field_results = self._compare_fields(json_data, ground_truth_row)
        
        # Calculate statistics
        total_fields = len(field_results)
        fields_right = sum(1 for r in field_results if r.judgment == Judgment.RIGHT)
        fields_wrong = sum(1 for r in field_results if r.judgment == Judgment.WRONG)
        fields_skipped = sum(1 for r in field_results if r.judgment == Judgment.SKIPPED)
        
        evaluated_fields = fields_right + fields_wrong
        accuracy = (fields_right / evaluated_fields * 100) if evaluated_fields > 0 else 100.0
        
        report = EvaluationReport(
            evaluation_id=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{statement_id[:8]}",
            json_file=json_path,
            statement_id=statement_id,
            evaluation_timestamp=datetime.now().isoformat(),
            processing_time_seconds=round((datetime.now() - start_time).total_seconds(), 3),
            total_fields_evaluated=total_fields,
            fields_right=fields_right,
            fields_wrong=fields_wrong,
            fields_skipped=fields_skipped,
            accuracy_percentage=round(accuracy, 2),
            field_results=field_results,
            metadata={
                "evaluation_mode": "batched_single_call",
                "fields_with_mismatches": [
                    r.field_name for r in field_results if r.judgment == Judgment.WRONG
                ]
            }
        )
        
        self.logger.info_msg(
            f"Evaluation completed: {fields_right}/{evaluated_fields} fields correct "
            f"({accuracy:.2f}%) | Statement ID: {statement_id}"
        )
        
        if save_report:
            output_file = os.path.join(self.output_dir, f"evaluation_report_{statement_id}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        
        return report
    
    def batch_evaluate(
        self, 
        json_paths: List[str],
        continue_on_error: bool = True
    ) -> Dict[str, Any]:
        """Evaluate multiple JSON files (each uses a single LLM call)."""
        self.logger.info_msg(f"Starting batch evaluation of {len(json_paths)} files")
        
        batch_results = []
        successful = 0
        failed = 0
        failed_files = []
        document_accuracies = []
        
        for i, json_path in enumerate(json_paths, 1):
            self.logger.debug_msg(f"Processing [{i}/{len(json_paths)}]: {json_path}")
            
            try:
                report = self.evaluate(json_path, save_report=True)
                batch_results.append(report.to_dict())
                successful += 1
                document_accuracies.append(report.accuracy_percentage)
            except Exception as e:
                self.logger.error_msg(f"Failed to evaluate {json_path}: {str(e)}")
                failed += 1
                failed_files.append({"file": json_path, "error": str(e)})
                if not continue_on_error:
                    raise
        
        overall_accuracy = (
            sum(document_accuracies) / len(document_accuracies) 
            if document_accuracies else 0.0
        )
        
        batch_report = {
            "batch_evaluation_id": f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "evaluation_timestamp": datetime.now().isoformat(),
            "evaluation_mode": "batched_single_call_per_document",
            "summary": {
                "total_documents": len(json_paths),
                "successful_evaluations": successful,
                "failed_evaluations": failed,
                "overall_accuracy_percentage": round(overall_accuracy, 2),
                "total_llm_calls": successful,  # One call per document!
            },
            "failed_files": failed_files,
            "individual_reports": batch_results,
        }
        
        batch_report_file = os.path.join(self.output_dir, "batch_evaluation_summary.json")
        with open(batch_report_file, 'w', encoding='utf-8') as f:
            json.dump(batch_report, f, indent=2, ensure_ascii=False)
        
        self.logger.info_msg(
            f"Batch evaluation completed: {successful}/{len(json_paths)} successful | "
            f"Overall accuracy: {overall_accuracy:.2f}% | Total LLM calls: {successful}"
        )
        
        return batch_report
    
    def clear_cache(self) -> None:
        """Clear the document cache."""
        self._document_cache.clear()
    
    def reset_agent(self) -> None:
        """Reset the agent instance."""
        self.agent = self._create_agent()