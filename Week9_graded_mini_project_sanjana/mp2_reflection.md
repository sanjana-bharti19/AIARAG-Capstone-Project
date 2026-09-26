# MP2 Reflection

## What worked

The part that really clicked for me was the section-aware chunking. Once I started carrying a clear `section` label with each chunk, retrieval felt much more grounded and the answers were easier to trace back to the right part of the story. It made a difference in cases like *The Adventure of the Speckled Band* and *A Scandal in Bohemia*, where the answer depends on a very specific clue or a narrow scene rather than the broader plot. Being able to tie each answer back to `source` + `title` + `section` also made the citations feel much more trustworthy, because I could tell whether the model was actually using the relevant passage instead of wandering into a general summary.

## What didn't work

The main problem I hit was metadata consistency. In the first version of the answer step, I was storing the human-readable story title as the citation `source`, while the validation harness was comparing against the exact filename such as `01_red_headed_league.txt`. That mismatch meant the system was citing something like “The Red-Headed League” while the expected source was `01_red_headed_league.txt`, so the source-match check failed even when the retrieved passage was correct. It was a good reminder that the validator is strict about exact source IDs, not just the story name.

Another technical issue came from the Qdrant client itself. The old `search()` method no longer exists in the installed library version, and the script crashed with an `AttributeError` because the client API had moved to `query_points()`. That was a real debugging dead end at first, because the code looked logically correct but the library contract had changed underneath it. Updating the retrieval call to the newer `query_points()` API fixed the crash, but it also reinforced how important it is to match the actual dependency version when building vector search pipelines.

## What I'd change

If I had another five hours, I would spend them on chunk quality and retrieval discipline rather than adding more complexity to the model pipeline. I’d tune the chunk sizes more carefully and add a lightweight reranking step so the system prefers the most relevant section instead of just the nearest vector match. I’d also tighten the answer prompt so the model has to cite the exact section and explicitly say when it does not have enough evidence, instead of improvising from vague context. I think that would help a lot more than trying to make the LLM smarter in isolation.

## One surprise

One thing that genuinely surprised me was how well the embedding model handled unusual names and distinctive clues even with a fairly simple setup. I did not expect `text-embedding-3-small` to be this strong at finding matches like “bell-rope,” “Egria,” or “annuity,” but it consistently picked up the relevant passage when the wording was close to the source. The other surprise was that the hardest part was not generation — it was metadata discipline. Once the `source` and `section` information was clean, the outputs felt much more reliable, and the quality jump was immediate.
