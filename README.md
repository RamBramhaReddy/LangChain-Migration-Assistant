# LangChain Migration Assistant

A RAG-powered tool that helps developers migrate their LangChain code from older versions (v0.1 through v1.0) to the latest APIs. Paste a code snippet, an error message, or just ask a question in plain English, and it finds the relevant migration documentation and generates a version-aware answer, telling you what changed, why, and how to update your code.

## What it does

LangChain has gone through several major API changes over the years — `initialize_agent()` became `create_agent()`, `chain.run()` became `chain.invoke()`, imports moved between packages, and so on. Keeping up with all of this by hand is tedious and error-prone.

This project builds a small RAG (Retrieval-Augmented Generation) system on top of LangChain's own documentation and release notes, so it can answer migration questions grounded in the actual docs instead of guessing.

You can:
- Paste a snippet of old LangChain code and get back the migrated version
- Paste an error message or stack trace and get an explanation plus a fix
- Ask a plain-English question like "What replaced `RetrievalQA`?" and get a direct answer

## How it works

**1. Data collection.** `fetch_langchain_data.py` pulls migration guides and release notes directly from LangChain's GitHub repos, across five version checkpoints (v0.1, v0.2, v0.3, v1.0, and latest).

**2. Cleaning.** `clean_docs.py` strips HTML tags, markdown links, emojis, and other noise out of the raw docs so they embed cleanly.

**3. Chunking and indexing.** `build_vector_db.py` splits the cleaned docs into chunks, tags each chunk with its version and doc type, and stores them in a Qdrant vector database. It also scans the docs for every backtick-wrapped API name it can find and saves them to `deprecated_apis.json` — this becomes a lookup table the retrieval step uses later.

**4. Input analysis.** When a question comes in, `code_detector.py` figures out what kind of input it is: is this actual Python code (checked by trying to parse it with Python's `ast` module), is it an error message, does it mention a specific version, and which API names appear in it.

**5. Hybrid retrieval.** `retrieval.py` combines two search strategies — dense vector search (via Qdrant's MMR search) and BM25 keyword search — merges the results, and then reranks everything with a cross-encoder model to pick the 5 most relevant chunks. This hybrid approach catches things that pure vector search alone tends to miss, like exact API names.

**6. Hallucination guard.** Before generating an answer, the pipeline checks whether the specific API the user asked about actually appears in the retrieved documentation. If it doesn't, the assistant says so honestly instead of making something up.

**7. Answer generation.** `prompts.py` builds a prompt (different structure depending on whether the input was code, an error, or a question) and sends it to an LLM (Groq's `openai/gpt-oss-120b` by default, with Mistral as a swappable alternative) along with the retrieved context.

**8. UI.** `app.py` is a Streamlit front end where you can paste your code or question and see the migration guide rendered nicely, including the original source documents it pulled from.

Every step is traced with Langfuse, so you can see exactly what was retrieved and generated for any given query.

## Project structure

```
├── app.py                     Streamlit web UI
├── main.py                    Simple CLI version
├── pipeline.py                Orchestrates the full RAG pipeline
├── retrieval.py                Hybrid search (dense + BM25) + reranking + hallucination guard
├── code_detector.py            Figures out what kind of input the user gave
├── prompts.py                  Builds the LLM prompt for each input type
├── fetch_langchain_data.py     Pulls docs from GitHub
├── clean_docs.py               Cleans raw markdown docs
├── build_vector_db.py          Chunks docs, builds the vector store, extracts API list
├── deprecated_apis.json        Auto-generated list of API names found in the docs
├── chunks.pkl                  Pickled document chunks (used for BM25 search)
├── qdrant_db/                  Local vector database (not committed — regenerate it, see below)
├── data/                       Raw and cleaned documentation (not committed — regenerate it, see below)
└── evaluation/
    ├── dataset.csv              30 test questions with ground-truth answers
    ├── generated_answers.csv    The pipeline's actual answers to those questions
    ├── manual_scoring.csv       Hand-scored evaluation results (see below)
    ├── fix_broken_rows.py       Small script used to re-run specific questions after a bug fix
    └── run_ragas.py             Automated evaluation using the Ragas library (see note below)
```

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Set up your `.env` file** with the API keys the project needs:

```
GROQ_API_KEY=your_key_here
MISTRAL_API_KEY=your_key_here
GITHUB_TOKEN=your_key_here
LANGFUSE_PUBLIC_KEY=your_key_here
LANGFUSE_SECRET_KEY=your_key_here
LANGFUSE_BASE_URL=your_langfuse_host
```

`GITHUB_TOKEN` is optional but recommended — without it you're limited to 60 GitHub API requests per hour, which isn't enough to fetch all the documentation.

**3. Build the knowledge base** (this only needs to be done once, or whenever you want to refresh the docs):

```bash
python fetch_langchain_data.py
python clean_docs.py
python build_vector_db.py
```

This downloads the docs, cleans them, and builds the local vector database. It creates the `data/` and `qdrant_db/` folders along with `chunks.pkl` and `deprecated_apis.json`.

**4. Run it**

Web UI:
```bash
streamlit run app.py
```

Command line:
```bash
python main.py
```

## Evaluation

To check how well the assistant actually performs, I put together a test set of 30 realistic migration questions covering imports, method renames, chain replacements, and agent construction, each with a hand-written ground-truth answer (`evaluation/dataset.csv`).

### What I tried first: automated evaluation with Ragas

I initially set up `evaluation/run_ragas.py` to score the answers automatically using [Ragas](https://github.com/explodinggradients/ragas), measuring context precision, context recall, faithfulness, and answer relevancy. The script includes batching, retry-with-backoff, and resume support, because each question requires several LLM calls per metric — with all four metrics, 30 questions works out to 100+ API calls.

In practice, this ran into free-tier rate limits (Groq and Mistral) that made a full run impractical without a paid API tier. The script itself works correctly (it got partway through a run before rate limits stopped it), but rather than burn a lot of time and API quota fighting rate limits for a resume project, I switched to manual evaluation instead — which turned out to be worth it, for reasons below.

### What I actually used: manual evaluation

I went through all 30 generated answers by hand and scored each one on:
- **Correctness** (0–5): does the answer actually match the ground truth?
- **Retrieval relevance** (yes/no): did the retrieved documents actually contain the right information?
- **Hallucination** (yes/no): did the answer invent something not present in the retrieved context?
- **Completeness** (yes/no): did it fully answer the question?

Results are in `evaluation/manual_scoring.csv`.

**Results:**

| Metric | Score |
|---|---|
| Average correctness | 3.60 / 5 (72%) |
| Fully correct answers | 16 / 30 (53%) |
| Retrieval found relevant docs | 22 / 30 (73%) |
| Hallucinations | 0 / 30 |

The zero hallucination rate is the result I'm most pleased with — it's the direct payoff of the hallucination guard in `retrieval.py`, which refuses to answer rather than make something up when the specific API isn't found in the retrieved context.

### Bugs the manual review caught

Going through the answers by hand surfaced two real bugs in the pipeline that raised the correctness score from 56.7% to 72% once fixed:

1. **Case-sensitive, exact-string hallucination guard.** The guard checked whether the exact requested API string (like `chain.run()`) appeared verbatim in the retrieved text. But the docs would say `Chain.run` (different case, no parentheses), so the guard would reject perfectly good retrieved context and refuse to answer. Fixed by normalizing case and stripping parentheses before comparing.

2. **A greedy JSON-extraction regex that hijacked good answers.** When the LLM's response wasn't valid JSON on its own, the code fell back to searching the entire response for anything that looked like a `{...}` block. Occasionally, the LLM's answer included a JSON-shaped code example (like a sample API payload) copied from the retrieved documentation, and the regex would grab that instead of recognizing the real answer was markdown, silently throwing away a correct response and replacing it with irrelevant JSON. Fixed by removing that fallback — the code now trusts its markdown fallback path (which was already there and working correctly) instead.

Both were the kind of thing that's easy to miss looking at aggregate scores alone, but obvious once you read a handful of actual outputs next to their expected answers. This is a big part of why the manual pass turned out to be more valuable than I originally expected — it wasn't just a fallback plan, it directly found the two most impactful bugs in the project.

### A known limitation

One question — "What replaced `ChatOpenAI.predict()`?" — is correctly declined by the hallucination guard. `ChatOpenAI` inherits `predict()` from `BaseChatModel`, and the documentation only ever refers to the deprecation as `BaseChatModel.predict`, never spelling out `ChatOpenAI.predict` directly. The guard has no notion of class inheritance, so it can't connect the two. This is an honest limitation rather than a bug — the alternative would be guessing, which is exactly what the guard is there to prevent.

## Notes on the tech stack

- **Retrieval:** Qdrant (dense vector search) + BM25 (keyword search) + a cross-encoder reranker, combined for hybrid retrieval
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **LLM:** Groq's `openai/gpt-oss-120b` (Mistral Large is available as a drop-in swap in `pipeline.py`)
- **Observability:** Langfuse traces every step of the pipeline
- **Evaluation:** Manual scoring (primary) + Ragas automated scoring (partial, infrastructure built but not fully run due to free-tier API limits)
- **UI:** Streamlit
