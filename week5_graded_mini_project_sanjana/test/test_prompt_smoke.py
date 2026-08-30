import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.prompt as prompt


async def validate_prompt_pipeline() -> None:
    """Run one real prompt call and assert the pipeline behaves as expected."""
    snippets, golden = prompt.load_data()
    assert snippets, "No job snippets were loaded from the dataset."
    assert golden, "No golden labels were loaded from the dataset."

    sample = snippets[0]
    result = await prompt.run_one("few_shot", sample)

    assert result["strategy"] == "few_shot", "Wrong strategy was executed."
    assert result["snippet_id"] == sample["id"], "Snippet ID mismatch."
    assert isinstance(result["parsed_extraction"], dict), (
        "The model response did not parse into a JSON object."
    )

    expected_keys = {"company", "role", "years_experience_required"}
    missing = expected_keys - set((result["parsed_extraction"] or {}).keys())
    assert not missing, f"Missing expected keys: {sorted(missing)}"

    accuracy = prompt.score_accuracy(result["parsed_extraction"], golden[sample["id"]])
    assert 0 <= accuracy <= 3, f"Accuracy score out of range: {accuracy}"

    judge_score = await prompt.score_llm_judge(
        sample["snippet"], result["parsed_extraction"], golden[sample["id"]]
    )
    assert 1 <= judge_score <= 4, f"Judge score out of range: {judge_score}"

    print(f"Smoke test passed for snippet {sample['id']}.")
    print(f"Parsed extraction: {result['parsed_extraction']}")
    print(f"Accuracy: {accuracy}/3")
    print(f"Judge score: {judge_score}/4")


if __name__ == "__main__":
    asyncio.run(validate_prompt_pipeline())
