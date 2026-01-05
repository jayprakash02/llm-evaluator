import LLMEvaluator
# Usage example
if __name__ == "__main__":
    # Example configuration
    CSV_PATH = "data/curated.statements.csv"
    JSON_PATH = "data/214f42ca-2349-4919-a11e-540b65f4ab85-extract.json"
    OUTPUT_DIR = "evaluation_reports"
    
    print("=" * 60)
    print("LLM Evaluator for Invoice Data Extraction")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = LLMEvaluator(
        csv_path=CSV_PATH,
        output_dir=OUTPUT_DIR,
        enable_caching=True
    )
    
    # Evaluate a single JSON file
    try:
        report = evaluator.evaluate(JSON_PATH)
        
        # Print summary
        print(f"\n{'=' * 60}")
        print(f"EVALUATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Statement ID: {report.statement_id}")
        print(f"Total Fields Evaluated: {report.total_fields_evaluated}")
        print(f"Fields RIGHT: {report.fields_right}")
        print(f"Fields WRONG: {report.fields_wrong}")
        print(f"Fields SKIPPED: {report.fields_skipped}")
        print(f"Accuracy: {report.accuracy_percentage}%")
        print(f"Processing Time: {report.processing_time_seconds}s")
        print(f"\nReport saved to: {OUTPUT_DIR}/evaluation_report_{report.statement_id}.json")
        
        # Show mismatches
        mismatches = [r for r in report.field_results if r.judgment == Judgment.WRONG]
        if mismatches:
            print(f"\n{'=' * 60}")
            print("FIELD MISMATCHES")
            print(f"{'=' * 60}")
            for m in mismatches:
                print(f"  {m.field_name}:")
                print(f"    Extracted:    '{m.extracted_value}'")
                print(f"    Ground Truth: '{m.ground_truth_value}'")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the data files exist at the specified paths.")
    except ValueError as e:
        print(f"Validation Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")
        raise