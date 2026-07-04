"""
Automated RAG evaluation using Ragas (faithfulness, answer_relevancy).
context_precision and context_recall were dropped -- they fire one
sub-call per retrieved context chunk, roughly doubling total API calls
per question, and were the main driver of rate-limit failures on free
tiers.

STATUS: Built and validated on a partial run (batching, retry-on-429,
resume support all work correctly). Full 30-question evaluation was not
completed because the free-tier LLM APIs used here (Groq, Mistral) enforce
rate limits (RPM/TPM/RPD) far too low for Ragas's call volume -- each
question fires several sub-calls per metric, so even with just two
metrics, 30 questions add up quickly against free-tier quotas.

DECISION: Given the free-tier constraint, this project's final evaluation
was done manually instead -- see evaluation/manual_scoring_completed.csv
and README for methodology (per-question correctness, retrieval
relevance, hallucination, and completeness scores, 0-5 scale). Manual
review also caught two real pipeline bugs (a case-sensitive substring
check in the hallucination guard, and a regex fallback that could hijack
a good markdown answer) that Ragas's aggregate scores alone would not
have surfaced as clearly.

This script is kept as a working reference for automated evaluation and
would be the natural next step with a paid API tier (higher RPM/TPM
removes the bottleneck; no code changes needed here to scale up).
"""

import os
import sys
import time
import json
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, PROJECT_ROOT)

from pipeline import run_pipeline

from datasets import Dataset

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model

from ragas import evaluate
from ragas.run_config import RunConfig

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
)

load_dotenv()

DATASET_PATH = "evaluation/dataset.csv"

RESULTS_PATH = "evaluation/results.csv"

GENERATED_ANSWERS_PATH = "evaluation/generated_answers.csv"

BATCH_SIZE = 3

BATCH_DELAY_SECONDS = 180

MAX_BATCH_RETRIES = 3

llm = init_chat_model(
    "openai/gpt-oss-120b",
    model_provider="groq"
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

questions = []
ground_truths = []
answers = []
contexts = []

if os.path.exists(GENERATED_ANSWERS_PATH):

    print("=" * 100)
    print("Loading Cached Answers")
    print("=" * 100)

    cached_df = pd.read_csv(
        GENERATED_ANSWERS_PATH,
        keep_default_na=False
    )

    cached_df["question"] = cached_df["question"].astype(str)
    cached_df["ground_truth"] = cached_df["ground_truth"].astype(str)
    cached_df["answer"] = cached_df["answer"].astype(str)

    empty_mask = cached_df["answer"].str.strip() == ""
    n_empty = empty_mask.sum()

    if n_empty:
        print()
        print(f"Skipping {n_empty} row(s) with empty answer:")
        for q in cached_df.loc[empty_mask, "question"]:
            print(f"  - {q}")

    cached_df = cached_df[~empty_mask].reset_index(drop=True)

    questions = cached_df["question"].tolist()
    ground_truths = cached_df["ground_truth"].tolist()
    answers = cached_df["answer"].tolist()

    contexts = []

    for context in cached_df["contexts"]:

        if pd.isna(context) or context == "":
            contexts.append([])
        else:
            contexts.append(json.loads(context))

    print()
    print(f"Loaded {len(questions)} cached answers from {GENERATED_ANSWERS_PATH}")

else:

    df = pd.read_csv(DATASET_PATH)

    print("=" * 100)
    print("Generating Answers")
    print("=" * 100)

    for i, row in df.iterrows():

        question = row["question"]
        ground_truth = row["ground_truth"]

        print()
        print(f"[{i + 1}/{len(df)}] {question}")

        try:

            result = run_pipeline(question)

            answer = result["summary"]

            retrieved_contexts = result["contexts"]

        except Exception as e:

            print(e)

            answer = ""

            retrieved_contexts = []

        questions.append(str(question) if question is not None else "")

        ground_truths.append(str(ground_truth) if ground_truth is not None else "")

        answers.append(str(answer) if answer is not None else "")

        contexts.append(retrieved_contexts)

        time.sleep(1)

    print()
    print("=" * 100)
    print("Generation Complete")
    print("=" * 100)

    generated_df = pd.DataFrame(
        {
            "question": questions,
            "ground_truth": ground_truths,
            "answer": answers,
            "contexts": [
                json.dumps(context)
                for context in contexts
            ]
        }
    )

    generated_df.to_csv(
        GENERATED_ANSWERS_PATH,
        index=False
    )

    print()
    print(f"Saved {GENERATED_ANSWERS_PATH}")

print()
print("=" * 100)
print("Running Ragas Evaluation In Batches")
print("=" * 100)

total = len(questions)

print(f"Total questions : {total}")
print(f"Batch size      : {BATCH_SIZE}")
print()

if os.path.exists(RESULTS_PATH):
    existing_results_df = pd.read_csv(RESULTS_PATH)
    done_questions = set(existing_results_df["question"].astype(str).tolist())
    print(f"Found existing results with {len(done_questions)} question(s) already scored")
else:
    existing_results_df = pd.DataFrame()
    done_questions = set()

all_batch_results = [existing_results_df] if not existing_results_df.empty else []

for batch_start in range(0, total, BATCH_SIZE):

    batch_end = min(batch_start + BATCH_SIZE, total)

    batch_questions = questions[batch_start:batch_end]
    batch_ground_truths = ground_truths[batch_start:batch_end]
    batch_answers = answers[batch_start:batch_end]
    batch_contexts = contexts[batch_start:batch_end]

    filtered_questions = []
    filtered_answers = []
    filtered_contexts = []
    filtered_ground_truths = []

    for q, a, c, g in zip(
        batch_questions,
        batch_answers,
        batch_contexts,
        batch_ground_truths,
    ):
        if q in done_questions:
            continue

        filtered_questions.append(q)
        filtered_answers.append(a)
        filtered_contexts.append(c)
        filtered_ground_truths.append(g)

    if not filtered_questions:
        print(f"Batch {batch_start + 1}-{batch_end} already complete, skipping")
        continue

    print()
    print("-" * 100)
    print(f"Batch {batch_start + 1}-{batch_end} of {total}")
    print(f"  {len(filtered_questions)} question(s) to score "
          f"({len(batch_questions) - len(filtered_questions)} already done)")
    print("-" * 100)

    batch_dataset = Dataset.from_dict(
        {
            "question": filtered_questions,
            "answer": filtered_answers,
            "retrieved_contexts": filtered_contexts,
            "reference": filtered_ground_truths,
        }
    )

    batch_df = None

    for retry in range(MAX_BATCH_RETRIES):

        try:

            batch_result = evaluate(
                dataset=batch_dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                ],
                llm=llm,
                embeddings=embeddings,
                run_config=RunConfig(
                    timeout=300,
                    max_retries=3,
                    max_wait=60,
                    max_workers=1
                ),
                raise_exceptions=False
            )

            batch_df = batch_result.to_pandas()

            break

        except Exception as e:

            msg = str(e).lower()

            if "429" in msg or "rate limit" in msg:

                wait = 120 * (retry + 1)

                print(f"Rate limited. Waiting {wait} seconds...")

                time.sleep(wait)

            else:

                print(e)

                batch_df = None

                break

    if batch_df is None:
        print(f"Batch {batch_start + 1}-{batch_end} failed, will retry on next run")
        continue

    all_batch_results.append(batch_df)

    if not batch_df.empty and "question" in batch_df.columns:
        done_questions.update(
            batch_df["question"].astype(str).tolist()
        )

    combined_so_far = pd.concat(all_batch_results, ignore_index=True)
    combined_so_far.to_csv(RESULTS_PATH, index=False)

    print(f"Saved progress: {len(combined_so_far)} row(s) so far")

    if batch_end < total:
        print(f"Waiting {BATCH_DELAY_SECONDS} seconds before next batch...")
        time.sleep(BATCH_DELAY_SECONDS)

print()
print("=" * 100)
print("Evaluation Complete")
print("=" * 100)

if os.path.exists(RESULTS_PATH):
    results_df = pd.read_csv(RESULTS_PATH)
else:
    print("No successful evaluation results were produced.")
    sys.exit(0)

print()
print("=" * 100)
print("Average Scores")
print("=" * 100)

skip = {
    "question",
    "answer",
    "reference"
}

for column in results_df.columns:

    if column in skip:
        continue

    try:
        print(f"{column:<25}: {results_df[column].mean():.4f}")
    except Exception:
        pass

print()
print("=" * 100)
print("Saved Results")
print("=" * 100)

print(f"Results File : {RESULTS_PATH}")

print()
print(results_df.head())