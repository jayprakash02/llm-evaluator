from LLMEvaluator import LLMEvaluator, Judgment, EvaluationReport
# Usage example
# Convenience function for quick evaluations
if __name__ == "__main__":
    CSV_PATH = "data/curated.statements.csv"
    JSON_PATH = "data/214f42ca-2349-4919-a11e-540b65f4ab85-extract.json"
    OUTPUT_DIR = "evaluation_reports"
    
    print("=" * 60)
    print("LLM Evaluator for Invoice Data Extraction")
    print("=" * 60)
    
    # Initialize evaluator (agent created once here)
    evaluator = LLMEvaluator(
        csv_path=CSV_PATH,
        output_dir=OUTPUT_DIR,
        enable_caching=True,
        log_level="DEBUG"  # Set to DEBUG for verbose output
    )
    
    # Single evaluation
    try:
        report = evaluator.evaluate(JSON_PATH)
        
        print(f"\n{'=' * 60}")
        print("EVALUATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Statement ID: {report.statement_id}")
        print(f"Accuracy: {report.accuracy_percentage}%")
        print(f"Fields: {report.fields_right} RIGHT, {report.fields_wrong} WRONG, {report.fields_skipped} SKIPPED")
        
    except Exception as e:
        print(f"Error: {e}")
        raise