# Evaluation Set

This directory contains the held-out evaluation dataset used to score prompt versions in `evaluate.py`.

## Structure

```
eval/
├── documents/       # Input documents (plain text)
│   ├── doc1.txt    # Quarterly sales report
│   ├── doc2.txt    # Research paper abstract
│   ├── doc3.txt    # Technology news article
│   ├── doc4.txt    # Terms of service update
│   └── doc5.txt    # Open-source library README
└── references/     # Human-written reference summaries (ground truth)
    ├── ref1.txt
    ├── ref2.txt
    ├── ref3.txt
    ├── ref4.txt
    └── ref5.txt
```

## How it works

`evaluate.py` runs each prompt version (v1, v2, v3) against every document. The generated `summary` field is compared against the corresponding reference using ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L). Results are logged to MLflow and saved to `eval_results/`.

## How to extend

To add a new evaluation document:

1. Add `doc{n}.txt` to `eval/documents/` — any plain text document
2. Add `ref{n}.txt` to `eval/references/` — a 2–3 sentence human-written summary capturing the key points
3. Run `python evaluate.py` to score all prompt versions against the updated set

Reference summaries should be written to capture the most important facts in the document, not to be stylistically similar to model output. The ROUGE score measures lexical overlap — references that use specific numbers, names, and key terms will produce more meaningful scores.

## Why the eval set is committed

The evaluation dataset is version-controlled alongside the code so that results are reproducible. Running `evaluate.py` on any machine with the same eval set and prompt versions should produce comparable scores. This is the same principle as committing test fixtures in software testing.
