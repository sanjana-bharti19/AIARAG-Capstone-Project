# MP3 Reflection

## Grounding and abstention

This project showed me that a good RAG system is not just about retrieval — it is about being careful about what the model is allowed to say. The assistant should answer only from the corpus, and if the evidence is missing or weak, it should abstain instead of guessing.

## Citations and traceability

Citations make the answer auditable. If the model cites a chunk that does not actually support the claim, the system should catch it. That is what makes the response grounded rather than just fluent.

## One surprise: the Q1 issue we encountered

The biggest surprise was Q1:

> Who was the assistant working at Jabez Wilson’s pawnshop, and what was his real identity?

This question has two linked facts: the alias, Vincent Spaulding, and the real identity, John Clay. A single retrieval query often found only one part of the answer, because the evidence was spread across different sections of the story.

We fixed this by improving chunking with overlap and by splitting compound questions into sub-questions, retrieving each one separately, and merging the results before answering. This made the system much more reliable for multi-hop questions.

## Evaluation results

The final evaluation was strong:

- Retrieval hit rate: 1.0
- Grounded & cited rate: 1.0
- Abstention accuracy: 1.0
- Hallucination rate: 0.0

This showed that the system retrieved the right sources, answered with valid citations, and correctly refused unsupported questions.

## Lessons learned

The main lesson is that trustworthy AI needs structure: good chunking, better retrieval coverage, citation validation, and a clear abstention policy. Without those safeguards, a model may sound confident while still being unsupported.

