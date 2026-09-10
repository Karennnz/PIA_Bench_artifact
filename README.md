# PIA-Bench Artifact

Data and materials accompanying *PIA-Bench: Towards Automated Privacy Impact Assessment with Large Language Models*.

## Scope

The release covers 499 publicly available Privacy Impact Assessments from five U.S. federal agencies. Structural screening retained 73 assessments from four agencies.

## Contents

- `sources/`: official agency catalogue pages and access dates.
- `data/screening/`: document-level structural-screening outputs.
- `data/units/`: retained documents, text units, and attribute annotations.
- `data/gold/`: extracted reference spans and extraction metadata.
- `data/generations/`: stored model generations and baseline predictions.
- `data/judge/`: item-level judge outputs.
- `data/scores/`: evaluation results at the reported cutoffs.
- `data/readability/`: Flesch Reading Ease results.
- `data/topics/`: topic codebooks, assignments, and figure source tables.
- `data/validation/`: validation materials.
- `prompts/`: generation and judging prompts.
- `scripts/`: pipeline code organized by processing stage.

Files are provided in JSON, JSONL, CSV, XLSX, TXT, and Python formats. Source PDFs are not redistributed; their official catalogue pages are listed in `sources/agency_source_pages.csv`. Some scripts require the corresponding model access, Python dependencies, and local path configuration.
