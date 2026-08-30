# MP1 Writeup

## 1) Which prompting strategy worked best?

The few-shot strategy performed best overall. It achieved the highest mean accuracy at 3.0 out of 3, and it tied for the strongest judge score at 3.3, while keeping a perfect parse rate of 1.0. The structured strategy was close behind at 2.8 accuracy, but the few-shot prompt was more consistent at extracting the required fields without hallucinating missing values. The zero-shot variant was the weakest at 2.4 accuracy, showing that giving the model examples materially improved reliability. For this task, the extra examples reduced ambiguity and made the expected output format clearer.

## 2) What did the comparison teach me about prompt design?

The results show that prompt design matters more than model choice in this task. All four strategies had the same parse rate of 1.0, which means all prompt variants were able to return valid JSON, but the content quality varied substantially. Few-shot prompting made the model more likely to match the required field names and values, especially for edge cases such as the minimum years requirement or missing data. The structured prompt improved clarity, but it still fell slightly behind few-shot in accuracy. This suggests that examples are often more helpful than a long specification when the task is a straightforward extraction format.

## 3) What surprised me or what was difficult?

The biggest challenge was handling ambiguous or partial information in the snippets. Some job postings included phrases like “3+ years” or “preferred,” which required the model to infer the intended minimum value rather than inventing data. The judge score was useful because it highlighted that a model might produce a structurally valid answer that is still not fully accurate. Cost and latency were also helpful for calibration: although all strategies were effectively $0.0 in this run, the latency differences were still visible, with cot at 1.816s and few-shot at 1.921s. This reminds me that prompt quality is not just about correctness; it is also about how well the system behaves under repeated use.

## 4) How would I improve this in a real pipeline?

I would keep the few-shot strategy as the baseline and add tighter normalization rules before scoring. For example, I would explicitly instruct the model to convert values like “5+” or “around 6 years” into integers and to return null when the requirement is not stated. I would also keep the parser simple and robust by removing markdown fences and validating the JSON object before aggregation. In a production pipeline, I would pair this with a small normalization step and a judge pass only when needed, which would improve both correctness and cost efficiency without overcomplicating the system.
