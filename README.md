# LLM Evaluator

## Context

We have a **[curated, human-verified CSV](data/curated.statements.csv)** that represents **ground truth** invoice data where:
  - each row represents a document/invoice
  - column headers are field keys (e.g., `account_number`, `amount_due`, etc.)
  - one of the columns is `statement_id`, which uniquely identifies the document

For each document, we also have an **extracted JSON** file (produced by the GroundX pipeline) containing key/value pairs that match the column headers in [curated.statements.csv](data/curated.statements.csv).

We need a Python class that takes a string path (URL or local path) to an extracted JSON file and uses an LLM to evaluate each field as **RIGHT** or **WRONG**, where “right/wrong” is defined by the corresponding values in the CSV.

An output containing the key-values from the JSON file and whether the values are right or wrong should be created. The format of the output (JSON, CSV, etc...) is up to you but your choice should be defensible.

## Goal / Success Criteria

Given:
- a curated CSV file (ground truth) at [data/curated.statements.csv](data/curated.statements.csv)
- an extracted JSON (local path or URL)

The evaluator must:
1. match the JSON to the correct CSV row
  - the JSON includes a `file_name` key-value
  - the JSON `file_name` should match the column `statement_id` in the CSV
    - the JSON `file_name` is in the format of `{statement_id}.pdf`
    - if you strip `.pdf`, the remainder of `file_name` should match a `statement_id` in the CSV
2. compare each relevant extracted field to the ground truth
3. call an LLM to judge each field as **RIGHT** or **WRONG** (handling formatting differences)
4. return a structured report and log key outcomes

## Data Sources

### 1) Curated Ground Truth CSV (Source of Truth)
- Can be found at at [data/curated.statements.csv](data/curated.statements.csv).
- Each row corresponds to one document.
- Column headers are the canonical field keys (e.g., `account_number`, `amount_due`, etc.)

**Ground truth definition:** the CSV value is the correct answer.

### 2) Extracted JSON (To Be Evaluated)
- A JSON file containing a single JSON object.
  - An example can be found at [data/214f42ca-2349-4919-a11e-540b65f4ab85-extract.json](data/214f42ca-2349-4919-a11e-540b65f4ab85-extract.json)


We need a Python implementation that compares extracted JSON values against curated CSV values and uses an LLM to identify and explain mismatches.

### Required Classes

The implementation **must** use:
- [`groundx.extract.AgentSettings`](https://github.com/eyelevelai/groundx-python/blob/9a5d334716dc0868f61e3902a24ef543a3c3e1b0/src/groundx/extract/settings/settings.py#L24) (LLM configuration)
- [`groundx.extract.services.Logger`](https://github.com/eyelevelai/groundx-python/blob/9a5d334716dc0868f61e3902a24ef543a3c3e1b0/src/groundx/extract/services/logger.py#L6) (logging)

