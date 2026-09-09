# Data

Controlled experiments are synthetic and require no external dataset.

For Amazon Reviews 2023, place `Home_and_Kitchen.jsonl.gz` beside the repository:

```text
parent/
├── Home_and_Kitchen.jsonl.gz
├── amazon_preprocess/
└── aggregate-reuse/
```

Run `experiments/amazon/preprocess.py` to construct the shared reusable
`../amazon_preprocess/` hierarchy. Downstream Amazon experiments read this
shared hierarchy without copying large intermediates into the repository.
Frozen preprocessing rules are in `configs/amazon_preprocess.yaml`.
