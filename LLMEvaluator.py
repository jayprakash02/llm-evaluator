import json
import csv
import os
from typing import Dict, List, Any
import openai  # or your chosen LLM client

class LLMEvaluator:
    """
    Evaluates extracted JSON invoice data against a curated CSV ground truth.
    """
    def __init__(self, csv_path: str):
        """
        Initialize with the path to the ground truth CSV.
        Loads the CSV into memory for quick lookup.
        """
        self.ground_truth = {}
        with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Use statement_id as the unique key for each document
                self.ground_truth[row['statement_id']] = row
        # Optional: Initialize LLM client here
        # self.llm_client = ...

    def evaluate_json(self, json_path: str) -> Dict[str, Any]:
        """
        Main method to evaluate a single extracted JSON file.
        """
        # 1. Load the extracted JSON
        with open(json_path, 'r', encoding='utf-8') as file:
            extracted_data = json.load(file)

        # 2. Match to ground truth using file_name
        file_name = extracted_data.get('file_name', '')
        statement_id = os.path.splitext(file_name)[0]  # Remove '.pdf'

        if statement_id not in self.ground_truth:
            raise ValueError(f"No ground truth found for statement_id: {statement_id}")

        truth_row = self.ground_truth[statement_id]
        report = []

        # 3. Compare each relevant field
        for field_key, extracted_value in extracted_data.items():
            # Skip the file_name key itself for field-by-field comparison
            if field_key == 'file_name' or field_key not in truth_row:
                continue

            truth_value = truth_row[field_key]

            # 4. Use LLM to judge RIGHT/WRONG
            judgment = self._get_llm_judgment(field_key, truth_value, extracted_value)

            report.append({
                "field": field_key,
                "ground_truth": truth_value,
                "extracted": extracted_value,
                "judgment": judgment
            })

        # 5. Return structured report
        return {
            "statement_id": statement_id,
            "file_name": file_name,
            "field_evaluations": report
        }

    def _get_llm_judgment(self, field: str, truth: str, extracted: str) -> str:
        """
        Queries an LLM to determine if the extracted value is RIGHT or WRONG.
        This is a placeholder. You must implement the actual API call.
        """
        # Construct a precise prompt
        prompt = f"""
        Compare the extracted value against the ground truth for the field '{field}'.
        Consider formatting differences (spaces, casing, date formats) as acceptable.

        Ground Truth: "{truth}"
        Extracted Value: "{extracted}"

        If the extracted value is semantically correct and matches the ground truth,
        respond with only the word "RIGHT".
        If the extracted value is incorrect or does not match, respond with only the word "WRONG".
        Do not include any other text or explanation.
        """
        # Example for OpenAI (requires `openai` library and API key)
        # response = openai.chat.completions.create(
        #     model="gpt-4o-mini",
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=0.0
        # )
        # judgment = response.choices[0].message.content.strip()
        # return judgment

        # Placeholder logic (replace with actual LLM call)
        if truth.strip().lower() == extracted.strip().lower():
            return "RIGHT"
        else:
            # Simulate LLM judging formatting differences
            return "WRONG"