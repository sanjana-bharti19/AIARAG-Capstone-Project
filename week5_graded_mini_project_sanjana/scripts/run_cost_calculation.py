import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings

model = "gpt-4o-mini"
prompt_tokens = 1000
completion_tokens = 500

cost = (
    prompt_tokens * settings.OPENAI_COST_RATES[model]["in"]
    + completion_tokens * settings.OPENAI_COST_RATES[model]["out"]
)

print(f"Estimated cost: ${cost:.8f} USD")