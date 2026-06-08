# Aivery LOCOMO Benchmark

Scripts to reproduce the LOCOMO benchmark results from the Aivery README.

## What you need

- An Aivery API key (`aiv_...`) — [sign up at aivery.systems](https://aivery.systems)
- An OpenAI API key — for the LLM judge step only (~$5 for the full 1,540-question set)
- Python 3.10+
- The mem0 evaluation harness (dataset + metrics + eval scripts)

**Note:** A **Pro tier** or higher is recommended for the full dataset due to ingestion throughput limits.

## Setup

```bash
# 1. Clone the mem0 eval harness
git clone https://github.com/mem0ai/mem0
cd mem0/evaluation

# 2. Copy the Aivery integration files into the harness
cp -r /path/to/aivery-sdk/benchmark/src/aivery src/aivery

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your keys
export AIVERY_BASE_URL=https://api.aivery.systems
export AIVERY_API_KEY=aiv_...
export CORTEX_BASE_URL=https://cortex.aivery.systems
export OPENAI_API_KEY=sk-...        # only needed for the eval/judge step
```

## Run

```bash
# Step 1 — Ingest all 10 LOCOMO conversations into Aivery
python -c "
from src.aivery.add import AiveryAdd
m = AiveryAdd(data_path='dataset/locomo10.json')
m.process_all_conversations()
"

# Step 2 — Answer all questions via Cortex (server-side LLM, no OpenAI needed here)
python -c "
from src.aivery.search import AiverySearch
s = AiverySearch(output_path='results/aivery_results.json', top_k=50)
s.process_data_file('dataset/locomo10.json')
"

# Step 3 — LLM judge evaluation (requires OPENAI_API_KEY, ~$5)
python evals.py \
  --input_file  results/aivery_results.json \
  --output_file results/aivery_eval.json

# Step 4 — Print scores
python generate_scores.py --input_file results/aivery_eval.json
```

## Cost breakdown

| Step | Who pays | Approx cost |
|---|---|---|
| Ingest (`add`) | Aivery plan (server-side extraction) | Covered by your tier |
| Search (`search`) | Aivery plan (Cortex answer generation) | Covered by your tier |
| Eval (`evals.py`) | Your OpenAI API key (LLM judge) | ~$5 for full dataset |

## Expected output

```
Mean Scores Per Category:
             llm_score  count
category
single-hop      0.74    282
temporal        0.73    321
multi-hop       0.66     96
open-domain     0.78    841

Overall Mean Scores:
llm_score    0.75
```
