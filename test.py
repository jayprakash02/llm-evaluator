"""
Unit Tests for LLM Evaluator

Run with: pytest test_llm_evaluator.py -v
Run with coverage: pytest test_llm_evaluator.py -v --cov=LLMEvaluator --cov-report=html
"""

import pytest
import json
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from LLMEvaluator import (
    LLMEvaluator,
    Judgment,
    FieldResult,
    EvaluationReport,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_csv_path():
    """Path to the actual test CSV file."""
    return "data/curated.statements.csv"


@pytest.fixture
def sample_json_paths():
    """List of all available test JSON files."""
    return [
        "data/214f42ca-2349-4919-a11e-540b65f4ab81-extract.json",
        "data/214f42ca-2349-4919-a11e-540b65f4ab82-extract.json",
        "data/214f42ca-2349-4919-a11e-540b65f4ab84-extract.json",
        "data/214f42ca-2349-4919-a11e-540b65f4ab85-extract.json",
        "data/214f42ca-2349-4919-a11e-540b65f4ab86-extract.json",
    ]


@pytest.fixture
def single_json_path():
    """Path to a single test JSON file."""
    return "data/214f42ca-2349-4919-a11e-540b65f4ab85-extract.json"


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp(prefix="llm_eval_test_")
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_agent_settings():
    """Create mock AgentSettings."""
    with patch('LLMEvaluator.AgentSettings') as mock_settings:
        settings = Mock()
        settings.model_id = "gpt-5-mini"
        settings.api_key = "test-api-key"
        settings.api_base = None
        settings.get_api_key.return_value = "test-api-key"
        settings.imports = ['json', 'pandas']
        mock_settings.return_value = settings
        yield settings


@pytest.fixture
def mock_agent():
    """Create a mock AgentCode that returns RIGHT/WRONG."""
    with patch('LLMEvaluator.AgentCode') as mock_agent_class:
        agent_instance = Mock()
        agent_instance.run.return_value = "RIGHT"
        mock_agent_class.return_value = agent_instance
        yield agent_instance


@pytest.fixture
def evaluator_with_mocks(sample_csv_path, temp_output_dir, mock_agent_settings, mock_agent):
    """Create an LLMEvaluator with mocked dependencies."""
    with patch('LLMEvaluator.AgentCode') as mock_agent_class:
        mock_agent_class.return_value = mock_agent
        evaluator = LLMEvaluator(
            csv_path=sample_csv_path,
            output_dir=temp_output_dir,
            enable_caching=True,
            log_level="DEBUG"
        )
        yield evaluator


# =============================================================================
# Test: Initialization
# =============================================================================

class TestLLMEvaluatorInit:
    """Tests for LLMEvaluator initialization."""
    
    def test_init_with_valid_csv(self, sample_csv_path, temp_output_dir, mock_agent):
        """Test successful initialization with valid CSV."""
        with patch('LLMEvaluator.AgentCode', return_value=mock_agent):
            evaluator = LLMEvaluator(
                csv_path=sample_csv_path,
                output_dir=temp_output_dir
            )
            
            assert evaluator.ground_truth_df is not None
            assert len(evaluator.ground_truth_df) == 5
            assert 'statement_id' in evaluator.ground_truth_df.columns
            assert evaluator.agent is not None
    
    def test_init_creates_output_dir(self, sample_csv_path, mock_agent):
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "new_output_dir")
            
            with patch('LLMEvaluator.AgentCode', return_value=mock_agent):
                evaluator = LLMEvaluator(
                    csv_path=sample_csv_path,
                    output_dir=output_dir
                )
                
                assert os.path.exists(output_dir)
    
    def test_init_with_nonexistent_csv_raises_error(self, temp_output_dir, mock_agent):
        """Test that FileNotFoundError is raised for missing CSV."""
        with patch('LLMEvaluator.AgentCode', return_value=mock_agent):
            with pytest.raises(FileNotFoundError, match="CSV file not found"):
                LLMEvaluator(
                    csv_path="nonexistent/path/file.csv",
                    output_dir=temp_output_dir
                )
    
    def test_init_normalizes_column_names(self, sample_csv_path, temp_output_dir, mock_agent):
        """Test that column names are normalized to lowercase."""
        with patch('LLMEvaluator.AgentCode', return_value=mock_agent):
            evaluator = LLMEvaluator(
                csv_path=sample_csv_path,
                output_dir=temp_output_dir
            )
            
            # All columns should be lowercase
            for col in evaluator.ground_truth_df.columns:
                assert col == col.lower()


# =============================================================================
# Test: CSV Loading and Validation
# =============================================================================

class TestCSVLoading:
    """Tests for CSV loading functionality."""
    
    def test_load_ground_truth_success(self, evaluator_with_mocks):
        """Test successful CSV loading."""
        df = evaluator_with_mocks.ground_truth_df
        
        assert df is not None
        assert len(df) == 5
        assert 'statement_id' in df.columns
    
    def test_statement_id_is_string(self, evaluator_with_mocks):
        """Test that statement_id column is converted to string."""
        df = evaluator_with_mocks.ground_truth_df
        
        assert df['statement_id'].dtype == object  # string type
    
    def test_empty_csv_raises_error(self, temp_output_dir, mock_agent):
        """Test that empty CSV raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("statement_id,field1,field2\n")  # Headers only
            temp_csv = f.name
        
        try:
            with patch('LLMEvaluator.AgentCode', return_value=mock_agent):
                with pytest.raises(ValueError, match="CSV file is empty"):
                    LLMEvaluator(csv_path=temp_csv, output_dir=temp_output_dir)
        finally:
            os.unlink(temp_csv)
    
    def test_csv_missing_statement_id_raises_error(self, temp_output_dir, mock_agent):
        """Test that CSV without statement_id column raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("field1,field2\n")
            f.write("value1,value2\n")
            temp_csv = f.name
        
        try:
            with patch('LLMEvaluator.AgentCode', return_value=mock_agent):
                with pytest.raises(ValueError, match="missing required 'statement_id' column"):
                    LLMEvaluator(csv_path=temp_csv, output_dir=temp_output_dir)
        finally:
            os.unlink(temp_csv)


# =============================================================================
# Test: JSON Loading
# =============================================================================

class TestJSONLoading:
    """Tests for JSON loading functionality."""
    
    def test_load_json_from_file(self, evaluator_with_mocks, single_json_path):
        """Test loading JSON from local file."""
        json_data = evaluator_with_mocks._load_json_data(single_json_path)
        
        assert json_data is not None
        assert isinstance(json_data, dict)
        assert 'file_name' in json_data
    
    def test_load_json_nonexistent_file_raises_error(self, evaluator_with_mocks):
        """Test that FileNotFoundError is raised for missing JSON."""
        with pytest.raises(FileNotFoundError, match="JSON file not found"):
            evaluator_with_mocks._load_json_data("nonexistent/file.json")
    
    def test_load_json_invalid_json_raises_error(self, evaluator_with_mocks):
        """Test that invalid JSON raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            temp_json = f.name
        
        try:
            with pytest.raises(ValueError, match="Invalid JSON"):
                evaluator_with_mocks._load_json_data(temp_json)
        finally:
            os.unlink(temp_json)


# =============================================================================
# Test: JSON to CSV Matching
# =============================================================================

class TestJSONToCSVMatching:
    """Tests for matching JSON files to CSV rows."""
    
    def test_match_json_to_csv_success(self, evaluator_with_mocks, single_json_path):
        """Test successful matching of JSON to CSV row."""
        json_data = evaluator_with_mocks._load_json_data(single_json_path)
        
        # Get statement_id from file_name
        expected_id = Path(json_data['file_name']).stem
        
        row = evaluator_with_mocks._match_json_to_csv(json_data)
        
        assert row is not None
        assert row['statement_id'].lower() == expected_id.lower()
    
    def test_match_json_missing_file_name_raises_error(self, evaluator_with_mocks):
        """Test that missing file_name raises ValueError."""
        json_data = {"field1": "value1"}  # No file_name
        
        with pytest.raises(ValueError, match="missing 'file_name' field"):
            evaluator_with_mocks._match_json_to_csv(json_data)
    
    def test_match_json_no_matching_row_raises_error(self, evaluator_with_mocks):
        """Test that unmatched statement_id raises ValueError."""
        json_data = {"file_name": "nonexistent-id.pdf"}
        
        with pytest.raises(ValueError, match="No CSV row found"):
            evaluator_with_mocks._match_json_to_csv(json_data)
    
    def test_match_json_case_insensitive(self, evaluator_with_mocks, single_json_path):
        """Test case-insensitive matching."""
        json_data = evaluator_with_mocks._load_json_data(single_json_path)
        
        # Modify to uppercase
        original_name = json_data['file_name']
        json_data['file_name'] = original_name.upper()
        
        # Should still match
        row = evaluator_with_mocks._match_json_to_csv(json_data)
        assert row is not None


# =============================================================================
# Test: Field Name Normalization
# =============================================================================

class TestFieldNameNormalization:
    """Tests for field name normalization."""
    
    def test_normalize_camel_case(self, evaluator_with_mocks):
        """Test camelCase to snake_case conversion."""
        assert evaluator_with_mocks._normalize_field_name("accountNumber") == "account_number"
        assert evaluator_with_mocks._normalize_field_name("amountDue") == "amount_due"
        assert evaluator_with_mocks._normalize_field_name("invoiceDate") == "invoice_date"
    
    def test_normalize_already_snake_case(self, evaluator_with_mocks):
        """Test that snake_case is preserved."""
        assert evaluator_with_mocks._normalize_field_name("account_number") == "account_number"
        assert evaluator_with_mocks._normalize_field_name("amount_due") == "amount_due"
    
    def test_normalize_uses_field_mappings(self, evaluator_with_mocks):
        """Test that explicit field mappings are used."""
        # These are defined in FIELD_MAPPINGS
        assert evaluator_with_mocks._normalize_field_name("accountNumber") == "account_number"
        assert evaluator_with_mocks._normalize_field_name("dueDate") == "due_date"


# =============================================================================
# Test: Field Value Cleaning
# =============================================================================

class TestFieldValueCleaning:
    """Tests for field value cleaning."""
    
    def test_clean_none_value(self, evaluator_with_mocks):
        """Test cleaning None values."""
        assert evaluator_with_mocks._clean_field_value(None) == ""
    
    def test_clean_nan_value(self, evaluator_with_mocks):
        """Test cleaning NaN values."""
        import math
        assert evaluator_with_mocks._clean_field_value(float('nan')) == ""
    
    def test_clean_whitespace(self, evaluator_with_mocks):
        """Test whitespace normalization."""
        assert evaluator_with_mocks._clean_field_value("  hello   world  ") == "hello world"
        assert evaluator_with_mocks._clean_field_value("line1\nline2") == "line1 line2"
    
    def test_clean_numeric_value(self, evaluator_with_mocks):
        """Test numeric value conversion."""
        assert evaluator_with_mocks._clean_field_value(123) == "123"
        assert evaluator_with_mocks._clean_field_value(123.45) == "123.45"


# =============================================================================
# Test: LLM Judgment
# =============================================================================

class TestLLMJudgment:
    """Tests for LLM judgment calls."""
    
    def test_call_llm_returns_right(self, evaluator_with_mocks):
        """Test LLM returning RIGHT judgment."""
        evaluator_with_mocks.agent.run.return_value = "RIGHT"
        
        result = evaluator_with_mocks._call_llm_for_judgment("test prompt")
        
        assert result == Judgment.RIGHT
    
    def test_call_llm_returns_wrong(self, evaluator_with_mocks):
        """Test LLM returning WRONG judgment."""
        evaluator_with_mocks.agent.run.return_value = "WRONG"
        
        result = evaluator_with_mocks._call_llm_for_judgment("test prompt")
        
        assert result == Judgment.WRONG
    
    def test_call_llm_handles_case_variations(self, evaluator_with_mocks):
        """Test LLM response case handling."""
        evaluator_with_mocks.agent.run.return_value = "right"
        assert evaluator_with_mocks._call_llm_for_judgment("test") == Judgment.RIGHT
        
        evaluator_with_mocks.agent.run.return_value = "Right"
        assert evaluator_with_mocks._call_llm_for_judgment("test") == Judgment.RIGHT
        
        evaluator_with_mocks.agent.run.return_value = "WRONG"
        assert evaluator_with_mocks._call_llm_for_judgment("test") == Judgment.WRONG
    
    def test_call_llm_handles_unexpected_response(self, evaluator_with_mocks):
        """Test handling of unexpected LLM responses."""
        evaluator_with_mocks.agent.run.return_value = "MAYBE"
        
        result = evaluator_with_mocks._call_llm_for_judgment("test prompt")
        
        assert result == Judgment.WRONG  # Default to WRONG
    
    def test_call_llm_handles_exception(self, evaluator_with_mocks):
        """Test handling of LLM exceptions."""
        evaluator_with_mocks.agent.run.side_effect = Exception("API Error")
        
        result = evaluator_with_mocks._call_llm_for_judgment("test prompt")
        
        assert result == Judgment.WRONG  # Default to WRONG on error


# =============================================================================
# Test: Caching
# =============================================================================

class TestCaching:
    """Tests for judgment caching."""
    
    def test_cache_stores_judgment(self, evaluator_with_mocks):
        """Test that judgments are cached."""
        evaluator_with_mocks.agent.run.return_value = "RIGHT"
        
        # First call
        prompt = evaluator_with_mocks._create_user_prompt("field1", "value1", "value1")
        evaluator_with_mocks._call_llm_for_judgment(prompt)
        
        # Check cache
        cache_key = evaluator_with_mocks._create_cache_key("field1", "value1", "value1")
        evaluator_with_mocks._judgment_cache[cache_key] = Judgment.RIGHT
        
        assert cache_key in evaluator_with_mocks._judgment_cache
    
    def test_clear_cache(self, evaluator_with_mocks):
        """Test cache clearing."""
        evaluator_with_mocks._judgment_cache["key1"] = Judgment.RIGHT
        evaluator_with_mocks._judgment_cache["key2"] = Judgment.WRONG
        
        evaluator_with_mocks.clear_cache()
        
        assert len(evaluator_with_mocks._judgment_cache) == 0
    
    def test_cache_key_uniqueness(self, evaluator_with_mocks):
        """Test that cache keys are unique for different inputs."""
        key1 = evaluator_with_mocks._create_cache_key("field", "a", "b")
        key2 = evaluator_with_mocks._create_cache_key("field", "b", "a")
        key3 = evaluator_with_mocks._create_cache_key("other", "a", "b")
        
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3


# =============================================================================
# Test: Field Comparison
# =============================================================================

class TestFieldComparison:
    """Tests for field comparison logic."""
    
    def test_exact_match_skips_llm(self, evaluator_with_mocks, single_json_path):
        """Test that exact matches skip LLM call."""
        json_data = evaluator_with_mocks._load_json_data(single_json_path)
        ground_truth = evaluator_with_mocks._match_json_to_csv(json_data)
        
        # Reset call count
        evaluator_with_mocks.agent.run.reset_mock()
        
        results = evaluator_with_mocks._compare_fields(json_data, ground_truth)
        
        # Check that exact matches didn't call LLM
        for result in results:
            if result.notes == "Exact match (case-insensitive)":
                assert result.judgment == Judgment.RIGHT
    
    def test_empty_values_skipped(self, evaluator_with_mocks):
        """Test that empty value pairs are skipped."""
        json_data = {"file_name": "test.pdf", "empty_field": ""}
        
        # Create mock ground truth with empty field
        ground_truth = pd.Series({
            "statement_id": "test",
            "empty_field": ""
        })
        
        # Mock _get_comparable_fields to return empty values
        with patch.object(evaluator_with_mocks, '_get_comparable_fields') as mock_get:
            mock_get.return_value = [("empty_field", "empty_field", "", "")]
            
            results = evaluator_with_mocks._compare_fields(json_data, ground_truth)
            
            assert len(results) == 1
            assert results[0].judgment == Judgment.SKIPPED


# =============================================================================
# Test: Single Evaluation
# =============================================================================

class TestSingleEvaluation:
    """Tests for single file evaluation."""
    
    def test_evaluate_returns_report(self, evaluator_with_mocks, single_json_path):
        """Test that evaluate returns an EvaluationReport."""
        evaluator_with_mocks.agent.run.return_value = "RIGHT"
        
        report = evaluator_with_mocks.evaluate(single_json_path, save_report=False)
        
        assert isinstance(report, EvaluationReport)
        assert report.statement_id is not None
        assert report.total_fields_evaluated >= 0
    
    def test_evaluate_saves_report(self, evaluator_with_mocks, single_json_path):
        """Test that report is saved when save_report=True."""
        evaluator_with_mocks.agent.run.return_value = "RIGHT"
        
        report = evaluator_with_mocks.evaluate(single_json_path, save_report=True)
        
        expected_file = os.path.join(
            evaluator_with_mocks.output_dir,
            f"evaluation_report_{report.statement_id}.json"
        )
        assert os.path.exists(expected_file)
    
    def test_evaluate_calculates_accuracy(self, evaluator_with_mocks, single_json_path):
        """Test accuracy calculation."""
        evaluator_with_mocks.agent.run.return_value = "RIGHT"
        
        report = evaluator_with_mocks.evaluate(single_json_path, save_report=False)
        
        # Accuracy should be between 0 and 100
        assert 0 <= report.accuracy_percentage <= 100
    
    def test_evaluate_nonexistent_file_raises_error(self, evaluator_with_mocks):
        """Test that missing JSON file raises error."""
        with pytest.raises(FileNotFoundError):
            evaluator_with_mocks.evaluate("nonexistent.json")


# =============================================================================
# Test: Batch Evaluation
# =============================================================================

class TestBatchEvaluation:
    """Tests for batch evaluation."""
    
    def test_batch_evaluate_all_files(self, evaluator_with_mocks, sample_json_paths):
        """Test batch evaluation of multiple files."""
        evaluator_with_mocks.agent.run.return_value = "RIGHT"
        
        # Filter to only files that exist and match CSV
        existing_paths = [p for p in sample_json_paths if os.path.exists(p)]
        
        result = evaluator_with_mocks.batch_evaluate(
            existing_paths,
            continue_on_error=True
        )
        
        assert 'batch_evaluation_id' in result
        assert 'summary' in result
        assert result['summary']['total_documents'] == len(existing_paths)
    
    def test_batch_evaluate_creates_summary_file(self, evaluator_with_mocks, sample_json_paths):
        """Test that batch summary file is created."""
        evaluator_with_mocks.agent.run.return_value = "RIGHT"
        
        existing_paths = [p for p in sample_json_paths if os.path.exists(p)][:2]
        
        evaluator_with_mocks.batch_evaluate(existing_paths)
        
        summary_file = os.path.join(
            evaluator_with_mocks.output_dir,
            "batch_evaluation_summary.json"
        )
        assert os.path.exists(summary_file)
    
    def test_batch_evaluate_continues_on_error(self, evaluator_with_mocks):
        """Test that batch continues on individual errors."""
        evaluator_with_mocks.agent.run.return_value = "RIGHT"
        
        paths = [
            "nonexistent1.json",
            "nonexistent2.json",
        ]
        
        result = evaluator_with_mocks.batch_evaluate(paths, continue_on_error=True)
        
        assert result['summary']['failed_evaluations'] == 2
        assert result['summary']['successful_evaluations'] == 0
    
    def test_batch_evaluate_stops_on_error_when_configured(self, evaluator_with_mocks):
        """Test that batch stops on error when continue_on_error=False."""
        with pytest.raises(FileNotFoundError):
            evaluator_with_mocks.batch_evaluate(
                ["nonexistent.json"],
                continue_on_error=False
            )


# =============================================================================
# Test: Data Classes
# =============================================================================

class TestDataClasses:
    """Tests for data classes."""
    
    def test_field_result_to_dict(self):
        """Test FieldResult serialization."""
        result = FieldResult(
            field_name="test_field",
            extracted_value="extracted",
            ground_truth_value="ground_truth",
            judgment=Judgment.RIGHT,
            notes="Test note"
        )
        
        d = result.to_dict()
        
        assert d['field_name'] == "test_field"
        assert d['judgment'] == "RIGHT"
        assert d['notes'] == "Test note"
    
    def test_evaluation_report_to_dict(self):
        """Test EvaluationReport serialization."""
        field_results = [
            FieldResult("f1", "e1", "g1", Judgment.RIGHT),
            FieldResult("f2", "e2", "g2", Judgment.WRONG),
        ]
        
        report = EvaluationReport(
            evaluation_id="test_id",
            json_file="test.json",
            statement_id="test_statement",
            evaluation_timestamp="2024-01-01T00:00:00",
            processing_time_seconds=1.5,
            total_fields_evaluated=2,
            fields_right=1,
            fields_wrong=1,
            fields_skipped=0,
            accuracy_percentage=50.0,
            field_results=field_results
        )
        
        d = report.to_dict()
        
        assert d['evaluation_id'] == "test_id"
        assert d['summary']['accuracy_percentage'] == 50.0
        assert 'f1' in d['field_details']
        assert 'f2' in d['field_details']


# =============================================================================
# Test: User Prompt Generation
# =============================================================================

class TestUserPromptGeneration:
    """Tests for user prompt generation."""
    
    def test_create_user_prompt_format(self, evaluator_with_mocks):
        """Test user prompt format."""
        prompt = evaluator_with_mocks._create_user_prompt(
            "account_number",
            "ACC-123",
            "ACC123"
        )
        
        assert "FIELD: account_number" in prompt
        assert 'EXTRACTED: "ACC-123"' in prompt
        assert 'GROUND TRUTH: "ACC123"' in prompt
        assert "RIGHT" in prompt
        assert "WRONG" in prompt


# =============================================================================
# Test: Integration Tests (with real data)
# =============================================================================

class TestIntegration:
    """Integration tests using real test data files."""
    
    @pytest.mark.integration
    def test_full_evaluation_flow(self, sample_csv_path, single_json_path, temp_output_dir, mock_agent):
        """Test complete evaluation flow with real files."""
        with patch('LLMEvaluator.AgentCode', return_value=mock_agent):
            mock_agent.run.return_value = "RIGHT"
            
            evaluator = LLMEvaluator(
                csv_path=sample_csv_path,
                output_dir=temp_output_dir,
                log_level="DEBUG"
            )
            
            report = evaluator.evaluate(single_json_path)
            
            assert report is not None
            assert report.statement_id is not None
            assert os.path.exists(
                os.path.join(temp_output_dir, f"evaluation_report_{report.statement_id}.json")
            )
    
    @pytest.mark.integration
    def test_batch_evaluation_with_real_files(self, sample_csv_path, sample_json_paths, temp_output_dir, mock_agent):
        """Test batch evaluation with real files."""
        with patch('LLMEvaluator.AgentCode', return_value=mock_agent):
            mock_agent.run.return_value = "RIGHT"
            
            evaluator = LLMEvaluator(
                csv_path=sample_csv_path,
                output_dir=temp_output_dir,
                log_level="DEBUG"
            )
            
            # Use only existing files
            existing = [p for p in sample_json_paths if os.path.exists(p)]
            
            result = evaluator.batch_evaluate(existing)
            
            assert result['summary']['successful_evaluations'] > 0


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_special_characters_in_values(self, evaluator_with_mocks):
        """Test handling of special characters."""
        value = evaluator_with_mocks._clean_field_value('Test "quoted" value')
        assert value == 'Test "quoted" value'
        
        value = evaluator_with_mocks._clean_field_value("Test's apostrophe")
        assert value == "Test's apostrophe"
    
    def test_unicode_characters(self, evaluator_with_mocks):
        """Test handling of unicode characters."""
        value = evaluator_with_mocks._clean_field_value("日本語テスト")
        assert value == "日本語テスト"
        
        value = evaluator_with_mocks._clean_field_value("Café résumé")
        assert value == "Café résumé"
    
    def test_very_long_values(self, evaluator_with_mocks):
        """Test handling of very long values."""
        long_value = "A" * 10000
        cleaned = evaluator_with_mocks._clean_field_value(long_value)
        assert len(cleaned) == 10000
    
    def test_numeric_precision(self, evaluator_with_mocks):
        """Test numeric value precision."""
        assert evaluator_with_mocks._clean_field_value(0.1 + 0.2) == "0.30000000000000004"
        assert evaluator_with_mocks._clean_field_value(1000000.99) == "1000000.99"


# =============================================================================
# Run Configuration
# =============================================================================

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "not integration"  # Skip integration tests by default
    ])