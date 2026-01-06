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
    
    # System prompt for the evaluation agent
    SYSTEM_PROMPT = """You are an expert evaluator for invoice data extraction accuracy.

Your task is to compare EXTRACTED VALUES against GROUND TRUTH VALUES and determine if they are semantically equivalent.

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

You must respond with EXACTLY one word: either "RIGHT" or "WRONG"
Do not include any explanation, punctuation, or additional text.
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
    
    # Fields to exclude from comparison (internal/metadata fields)
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
        """
        Initialize the LLM evaluator with ground truth data and AgentCode.
        
        The AgentCode is created once here with the system prompt and reused
        for all evaluations, making batch processing more efficient.
        
        Args:
            csv_path: Path to the curated CSV file containing ground truth data
            agent_settings: GroundX AgentSettings configuration for LLM interactions.
                           If None, default settings will be used.
            output_dir: Directory for saving evaluation reports
            enable_caching: Whether to cache LLM judgments for identical value pairs
            enable_tracing: Whether to enable OpenTelemetry tracing
            log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        
        Raises:
            FileNotFoundError: If the CSV file doesn't exist
            ValueError: If the CSV is empty or missing required columns
        """
        self.logger = Logger(__name__, log_level)
        self.logger.info_msg(f"Initializing LLMEvaluator with CSV: {csv_path}")
        
        # Validate and load ground truth data
        self._validate_csv_path(csv_path)
        self.ground_truth_df = self._load_ground_truth(csv_path)
        
        # Initialize GroundX agent settings
        self.agent_settings = agent_settings or AgentSettings()
        self.logger.debug_msg(
            f"AgentSettings - model_id: {self.agent_settings.model_id}, "
            f"api_base: {self.agent_settings.api_base}"
        )
        
        # Initialize the AgentCode once with system prompt
        self.agent = self._create_agent()
        self.logger.info_msg("AgentCode initialized with evaluation system prompt")
        
        # Configuration
        self.output_dir = output_dir
        self.enable_caching = enable_caching
        self._judgment_cache: Dict[str, Judgment] = {}
        
        # Create output directory if needed
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.debug_msg(f"Output directory: {self.output_dir}")

        if enable_tracing:
            self._setup_tracing()
    
    def _create_agent(self) -> AgentCode:
        """
        Create and configure the AgentCode instance with system prompt.
        
        Returns:
            Configured AgentCode instance
        """
        self.logger.debug_msg("Creating AgentCode instance...")
        
        agent = AgentCode(
            model_id=self.agent_settings.model_id,
            api_key=self.agent_settings.api_key or self.agent_settings.get_api_key(),
            api_base=self.agent_settings.api_base,
            system_prompt=self.SYSTEM_PROMPT,
        )
        
        self.logger.debug_msg(f"AgentCode created with model: {self.agent_settings.model_id}")
        return agent

    def _setup_tracing(self) -> None:
        """
        Setup OpenTelemetry tracing for LLM calls.
        
        Raises:
            ImportError: If required tracing packages are not installed
        """
        try:
            from openinference.instrumentation.smolagents import SmolagentsInstrumentor
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import (
                ConsoleSpanExporter,
                BatchSpanProcessor,
            )
            
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
        """
        Load and validate ground truth CSV data.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            DataFrame containing ground truth data
            
        Raises:
            ValueError: If CSV is empty or missing statement_id column
        """
        try:
            df = pd.read_csv(csv_path)
            
            # Normalize all column names to lowercase
            df.columns = df.columns.str.lower()
            
            self.logger.debug_msg(f"Loaded ground truth CSV with {len(df)} records")
            self.logger.debug_msg(f"CSV columns: {list(df.columns)}")
            
            if df.empty:
                raise ValueError("CSV file is empty")
            
            if 'statement_id' not in df.columns:
                raise ValueError("CSV missing required 'statement_id' column")
            
            # Ensure statement_id is string type for matching
            df['statement_id'] = df['statement_id'].astype(str)
            
            return df
            
        except pd.errors.EmptyDataError:
            raise ValueError(f"CSV file is empty: {csv_path}")
        except Exception as e:
            self.logger.error_msg(f"Failed to load CSV: {str(e)}")
            raise
    
    def _match_json_to_csv(self, json_data: Dict[str, Any]) -> pd.Series:
        """
        Match JSON file to the correct CSV row using file_name/statement_id.
        
        Args:
            json_data: Dictionary containing extracted JSON data
            
        Returns:
            Pandas Series containing the matching ground truth row
            
        Raises:
            ValueError: If no matching CSV row is found or file_name is missing
        """
        file_name = json_data.get('file_name', '')
        
        if not file_name:
            raise ValueError("JSON data missing 'file_name' field")
        
        # Extract statement_id by removing .pdf extension
        statement_id = Path(file_name).stem
        
        self.logger.debug_msg(f"Matching JSON: file_name='{file_name}' -> statement_id='{statement_id}'")
        
        # Find matching row in CSV
        matching_rows = self.ground_truth_df[
            self.ground_truth_df['statement_id'] == statement_id
        ]
        
        if len(matching_rows) == 0:
            # Try case-insensitive match
            matching_rows = self.ground_truth_df[
                self.ground_truth_df['statement_id'].str.lower() == statement_id.lower()
            ]
            
        if len(matching_rows) == 0:
            available_ids = self.ground_truth_df['statement_id'].head(5).tolist()
            error_msg = (
                f"No CSV row found for statement_id: '{statement_id}'. "
                f"Available IDs (first 5): {available_ids}"
            )
            self.logger.error_msg(error_msg)
            raise ValueError(error_msg)
        
        if len(matching_rows) > 1:
            self.logger.warning_msg(
                f"Multiple CSV rows ({len(matching_rows)}) found for "
                f"statement_id: '{statement_id}', using first match"
            )
        
        return matching_rows.iloc[0]
    
    def _normalize_field_name(self, field_name: str) -> str:
        """
        Normalize a field name to match CSV column conventions.
        
        Args:
            field_name: Original field name from JSON
            
        Returns:
            Normalized field name
        """
        # Check explicit mappings first
        if field_name in self.FIELD_MAPPINGS:
            return self.FIELD_MAPPINGS[field_name]
        
        # Convert camelCase to snake_case
        normalized = re.sub(r'(?<!^)(?=[A-Z])', '_', field_name).lower()
        
        return normalized
    
    def _clean_field_value(self, value: Any) -> str:
        """
        Clean and normalize field values for comparison.
        
        Args:
            value: Raw field value (could be string, number, NaN, None, etc.)
            
        Returns:
            Cleaned string representation
        """
        # Handle None and NaN
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        
        # Convert to string
        cleaned = str(value).strip()
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove common artifacts
        cleaned = cleaned.replace('\n', ' ').replace('\r', '')
        
        return cleaned
    
    def _create_cache_key(self, field_name: str, extracted: str, ground_truth: str) -> str:
        """Create a cache key for LLM judgment caching."""
        content = f"{field_name}|{extracted}|{ground_truth}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _create_user_prompt(
        self, 
        field_name: str, 
        extracted_value: str, 
        ground_truth_value: str
    ) -> str:
        """
        Create a user prompt for field evaluation.
        
        The system prompt already contains rules and examples, so this
        is a concise prompt for the specific field comparison.
        
        Args:
            field_name: Name of the field being evaluated
            extracted_value: Value extracted by the pipeline
            ground_truth_value: Ground truth value from CSV
            
        Returns:
            Formatted user prompt
        """
        prompt = f"""Evaluate this field extraction:

FIELD: {field_name}
EXTRACTED: "{extracted_value}"
GROUND TRUTH: "{ground_truth_value}"

Respond with only "RIGHT" or "WRONG":"""
        
        return prompt
    
    def _call_llm_for_judgment(self, user_prompt: str) -> Judgment:
        """
        Call the LLM agent to get field judgment.
        
        Uses the pre-initialized AgentCode with the system prompt.
        
        Args:
            user_prompt: The user prompt for this specific evaluation
            
        Returns:
            Judgment enum (RIGHT or WRONG)
        """
        try:
            self.logger.debug_msg(f"Calling LLM with prompt: {user_prompt[:100]}...")
            
            # Run the agent with user prompt (system prompt already configured)
            response = self.agent.run(user_prompt)
            
            # Handle response
            if isinstance(response, str):
                response_text = response
            elif hasattr(response, 'text'):
                response_text = response.text
            elif hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            self.logger.debug_msg(f"LLM response: {response_text}")
            
            # Extract judgment
            judgment_text = response_text.strip().upper()
            
            if "RIGHT" in judgment_text:
                return Judgment.RIGHT
            elif "WRONG" in judgment_text:
                return Judgment.WRONG
            else:
                self.logger.warning_msg(f"Unexpected LLM response: '{judgment_text}', defaulting to WRONG")
                return Judgment.WRONG
                
        except Exception as e:
            self.logger.error_msg(f"LLM call failed: {str(e)}")
            return Judgment.WRONG
    
    def _get_comparable_fields(
        self, 
        json_data: Dict[str, Any], 
        ground_truth: pd.Series
    ) -> List[Tuple[str, str, str, str]]:
        """
        Get list of fields that can be compared between JSON and CSV.
        
        Args:
            json_data: Extracted JSON data
            ground_truth: Ground truth row from CSV
            
        Returns:
            List of tuples (json_key, csv_column, extracted_value, ground_truth_value)
        """
        comparable_fields = []
        csv_columns = set(ground_truth.index) - {'statement_id'}
        json_keys = set(json_data.keys()) - self.EXCLUDED_FIELDS
        
        matched_csv_columns = set()
        
        for json_key in json_keys:
            # Try direct match
            if json_key in csv_columns:
                csv_col = json_key
            else:
                # Try normalized match
                normalized_key = self._normalize_field_name(json_key)
                if normalized_key in csv_columns:
                    csv_col = normalized_key
                else:
                    self.logger.debug_msg(f"JSON key '{json_key}' has no matching CSV column")
                    continue
            
            extracted_value = self._clean_field_value(json_data.get(json_key, ""))
            ground_truth_value = self._clean_field_value(ground_truth.get(csv_col, ""))
            
            comparable_fields.append((json_key, csv_col, extracted_value, ground_truth_value))
            matched_csv_columns.add(csv_col)
        
        # Log unmatched CSV columns at debug level
        unmatched_csv = csv_columns - matched_csv_columns
        if unmatched_csv:
            self.logger.debug_msg(f"CSV columns without JSON match: {unmatched_csv}")
        
        return comparable_fields
    
    def _compare_fields(
        self, 
        json_data: Dict[str, Any], 
        ground_truth: pd.Series
    ) -> List[FieldResult]:
        """
        Compare each relevant field between JSON and ground truth using LLM.
        
        Args:
            json_data: Extracted JSON data
            ground_truth: Ground truth row from CSV
            
        Returns:
            List of FieldResult objects
        """
        results = []
        comparable_fields = self._get_comparable_fields(json_data, ground_truth)
        
        self.logger.debug_msg(f"Comparing {len(comparable_fields)} fields")
        
        for json_key, csv_col, extracted_value, ground_truth_value in comparable_fields:
            field_name = csv_col
            
            # Skip comparison if both are empty
            if not extracted_value and not ground_truth_value:
                results.append(FieldResult(
                    field_name=field_name,
                    extracted_value=extracted_value,
                    ground_truth_value=ground_truth_value,
                    judgment=Judgment.SKIPPED,
                    notes="Both values empty - skipped comparison"
                ))
                self.logger.debug_msg(f"Field '{field_name}': SKIPPED (both empty)")
                continue
            
            # Check cache first
            cache_key = self._create_cache_key(field_name, extracted_value, ground_truth_value)
            if self.enable_caching and cache_key in self._judgment_cache:
                judgment = self._judgment_cache[cache_key]
                notes = "Cached judgment"
                self.logger.debug_msg(f"Field '{field_name}': {judgment.value} (cached)")
            else:
                # Quick check for exact match (skip LLM call)
                if extracted_value.lower() == ground_truth_value.lower():
                    judgment = Judgment.RIGHT
                    notes = "Exact match (case-insensitive)"
                    self.logger.debug_msg(f"Field '{field_name}': RIGHT (exact match)")
                else:
                    # Create user prompt and get judgment from LLM
                    user_prompt = self._create_user_prompt(field_name, extracted_value, ground_truth_value)
                    judgment = self._call_llm_for_judgment(user_prompt)
                    notes = "LLM semantic comparison"
                    self.logger.debug_msg(f"Field '{field_name}': {judgment.value} (LLM)")
                
                # Cache the result
                if self.enable_caching:
                    self._judgment_cache[cache_key] = judgment
            
            result = FieldResult(
                field_name=field_name,
                extracted_value=extracted_value,
                ground_truth_value=ground_truth_value,
                judgment=judgment,
                notes=notes
            )
            results.append(result)
            
            # Log mismatches at INFO level for visibility
            if judgment == Judgment.WRONG:
                self.logger.info_msg(
                    f"MISMATCH: {field_name} - "
                    f"extracted='{extracted_value}' vs ground_truth='{ground_truth_value}'"
                )
        
        return results
    
    def _load_json_data(self, json_path: str) -> Dict[str, Any]:
        """
        Load JSON data from local path or URL.
        
        Args:
            json_path: Local file path or URL to JSON file
            
        Returns:
            Dictionary containing JSON data
            
        Raises:
            ValueError: If JSON is invalid or cannot be loaded
            FileNotFoundError: If local file doesn't exist
        """
        if json_path.startswith(('http://', 'https://')):
            import requests
            try:
                self.logger.debug_msg(f"Fetching JSON from URL: {json_path}")
                response = requests.get(json_path, timeout=30)
                response.raise_for_status()
                json_data = response.json()
                self.logger.debug_msg(f"Loaded JSON from URL: {json_path}")
            except requests.exceptions.Timeout:
                raise ValueError(f"Timeout fetching URL: {json_path}")
            except requests.exceptions.RequestException as e:
                raise ValueError(f"Failed to fetch URL: {json_path} - {str(e)}")
        else:
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"JSON file not found: {json_path}")
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                self.logger.debug_msg(f"Loaded JSON from file: {json_path}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in file: {json_path} - {str(e)}")
        
        return json_data
    
    def evaluate(
        self, 
        json_path: str,
        save_report: bool = True
    ) -> EvaluationReport:
        """
        Evaluate a single JSON file against ground truth.
        
        Args:
            json_path: Path (local or URL) to the extracted JSON file
            save_report: Whether to save the report to a file
            
        Returns:
            EvaluationReport object containing all evaluation details
            
        Raises:
            FileNotFoundError: If JSON file doesn't exist
            ValueError: If JSON cannot be matched to CSV or is invalid
        """
        start_time = datetime.now()
        self.logger.info_msg(f"Starting evaluation for: {json_path}")
        
        # Load JSON data
        json_data = self._load_json_data(json_path)
        
        # Match JSON to CSV row
        ground_truth_row = self._match_json_to_csv(json_data)
        statement_id = Path(json_data.get('file_name', '')).stem
        
        # Compare all fields
        field_results = self._compare_fields(json_data, ground_truth_row)
        
        # Calculate summary statistics
        total_fields = len(field_results)
        fields_right = sum(1 for r in field_results if r.judgment == Judgment.RIGHT)
        fields_wrong = sum(1 for r in field_results if r.judgment == Judgment.WRONG)
        fields_skipped = sum(1 for r in field_results if r.judgment == Judgment.SKIPPED)
        
        # Calculate accuracy (excluding skipped fields)
        evaluated_fields = fields_right + fields_wrong
        accuracy = (fields_right / evaluated_fields * 100) if evaluated_fields > 0 else 100.0
        
        # Create report
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
                "ground_truth_csv_columns": list(self.ground_truth_df.columns),
                "extracted_json_keys": list(json_data.keys()),
                "fields_with_mismatches": [
                    r.field_name for r in field_results if r.judgment == Judgment.WRONG
                ]
            }
        )
        
        # Log summary
        self.logger.info_msg(
            f"Evaluation completed: {fields_right}/{evaluated_fields} fields correct "
            f"({accuracy:.2f}%) | Skipped: {fields_skipped} | Statement ID: {statement_id}"
        )
        
        # Save report if requested
        if save_report:
            output_file = os.path.join(
                self.output_dir, 
                f"evaluation_report_{statement_id}.json"
            )
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
            self.logger.debug_msg(f"Report saved to: {output_file}")
        
        return report
    
    def batch_evaluate(
        self, 
        json_paths: List[str],
        continue_on_error: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate multiple JSON files using the same agent instance.
        
        This is more efficient than creating separate evaluators because
        the AgentCode is reused across all evaluations.
        
        Args:
            json_paths: List of paths to JSON files
            continue_on_error: Whether to continue if individual evaluations fail
            
        Returns:
            Aggregated batch evaluation report
        """
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
        
        # Calculate overall statistics
        overall_accuracy = (
            sum(document_accuracies) / len(document_accuracies) 
            if document_accuracies else 0.0
        )
        
        # Create batch report
        batch_report = {
            "batch_evaluation_id": f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "evaluation_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_documents": len(json_paths),
                "successful_evaluations": successful,
                "failed_evaluations": failed,
                "overall_accuracy_percentage": round(overall_accuracy, 2),
                "min_accuracy": min(document_accuracies) if document_accuracies else None,
                "max_accuracy": max(document_accuracies) if document_accuracies else None,
                "document_accuracies": document_accuracies
            },
            "failed_files": failed_files,
            "individual_reports": batch_results,
            "output_directory": self.output_dir
        }
        
        # Save batch report
        batch_report_file = os.path.join(self.output_dir, "batch_evaluation_summary.json")
        with open(batch_report_file, 'w', encoding='utf-8') as f:
            json.dump(batch_report, f, indent=2, ensure_ascii=False)
        
        self.logger.info_msg(
            f"Batch evaluation completed: {successful}/{len(json_paths)} successful | "
            f"Overall accuracy: {overall_accuracy:.2f}%"
        )
        
        return batch_report
    
    def clear_cache(self) -> None:
        """Clear the LLM judgment cache."""
        cache_size = len(self._judgment_cache)
        self._judgment_cache.clear()
        self.logger.debug_msg(f"Cleared {cache_size} cached judgments")
    
    def reset_agent(self) -> None:
        """
        Reset the agent instance.
        
        Useful if you need to refresh the agent state between batches.
        """
        self.logger.debug_msg("Resetting AgentCode instance...")
        self.agent = self._create_agent()
        self.logger.debug_msg("AgentCode reset complete")

