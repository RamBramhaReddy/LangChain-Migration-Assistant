import os
import sys
import json
import pandas as pd

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, PROJECT_ROOT)

from pipeline import run_pipeline

GENERATED_ANSWERS_PATH = "evaluation/generated_answers.csv"

BROKEN_QUESTIONS = [
    "What replaced BaseLanguageModel.predict()?",
    "What replaced chain.run()?",
    "What replaced initialize_agent()?",
    "What replaced ChatOpenAI.predict()?",
    "agent = initialize_agent(tools, llm, agent='zero-shot-react-description')",
]

df = pd.read_csv(
    GENERATED_ANSWERS_PATH,
    keep_default_na=False
)

for question in BROKEN_QUESTIONS:

    mask = df["question"] == question

    if not mask.any():
        print(f"Not found in CSV, skipping: {question}")
        continue

    print()
    print("=" * 100)
    print(f"Re-running: {question}")
    print("=" * 100)

    try:

        result = run_pipeline(question)

        new_answer = result["summary"]
        new_contexts = result["contexts"]

    except Exception as e:

        print(f"Failed: {e}")
        continue

    df.loc[mask, "answer"] = new_answer
    df.loc[mask, "contexts"] = json.dumps(new_contexts)

    print(f"New answer (first 200 chars): {new_answer[:200]}")

df.to_csv(
    GENERATED_ANSWERS_PATH,
    index=False
)

print()
print(f"Saved updated {GENERATED_ANSWERS_PATH}")