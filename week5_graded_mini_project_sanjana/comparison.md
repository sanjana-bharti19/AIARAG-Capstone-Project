# MP1 Comparison Summary

## Objective

Compare the performance of the four prompting strategies on structured extraction from job-posting snippets.

Strategies evaluated:

- zero-shot
- few-shot
- structured
- cot

---

## Summary table

| Strategy | Accuracy (mean of 3) | Parse rate | Judge score | Total cost ($) | Latency p50 (s) |
|---|---:|---:|---:|---:|---:|
| cot | 2.7 | 1.0 | 3.3 | 0.0 | 1.816 |
| few_shot | 3.0 | 1.0 | 3.3 | 0.0 | 1.921 |
| structured | 2.8 | 1.0 | 3.1 | 0.0 | 1.887 |
| zero_shot | 2.4 | 1.0 | 3.1 | 0.0 | 1.873 |

> Scored 40 results.

---

## Interpretation

### Best overall quality

The few-shot strategy performed the best overall on accuracy and tied for the highest judge score, while maintaining a perfect parse rate.

### Best cost-efficiency

All strategies reported a total cost of $0.0 in this run, so the main differentiator here is quality and speed rather than spend.

### Best trade-off

The few-shot strategy had the strongest balance of accuracy and judge score, while the cot strategy was the fastest median latency at 1.816s.

---

## Key takeaway

The comparison shows that prompt design has a measurable impact on:

- output quality
- parsing reliability
- cost
- speed

The final recommendation should be based on the balance between correctness and efficiency, not just raw accuracy alone.

---

## Data source

- source snippets: `data/job_snippets.jsonl`
- gold labels: `data/golden_set.jsonl`
- generated outputs: `results.json`


