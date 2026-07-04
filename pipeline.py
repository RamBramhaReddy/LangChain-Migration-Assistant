import io
import json
import pickle
import re
from contextlib import redirect_stdout

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.chat_models import init_chat_model
from langchain_mistralai import ChatMistralAI

import os

from langfuse import Langfuse, observe

from code_detector import analyze_input
from retrieval import (
    retrieve_documents,
    hallucination_guard
)
from prompts import (
    build_prompt
)

load_dotenv()

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_BASE_URL")
)

if not langfuse.auth_check():
    raise RuntimeError(
        "Failed to authenticate with Langfuse. Check your API keys."
    )

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    path="./qdrant_db",
    collection_name="langchain_migration_docs"
)

with open("chunks.pkl", "rb") as f:
    all_chunks = pickle.load(f)

with open(
    "deprecated_apis.json",
    "r",
    encoding="utf-8"
) as f:
    deprecated_apis = json.load(f)

llm = init_chat_model(
    "openai/gpt-oss-120b",
    model_provider="groq"
)


# llm = ChatMistralAI(
#     model="mistral-large-latest",
#     api_key=os.getenv("MISTRAL_API_KEY"),
#     temperature=0
# )

REQUIRED_FIELDS = {
    "summary": "",
    "deprecated_apis": [],
    "updated_code": "",
    "notes": "",
    "example": ""
}


def _coerce_schema(parsed: dict) -> dict:
    """Fill any missing required keys so downstream code never KeyErrors."""
    result = dict(REQUIRED_FIELDS)
    if isinstance(parsed, dict):
        result.update(parsed)
    return result


def parse_llm_json(raw_text: str):
    """
    Try to parse raw_text as JSON.
    1. Direct json.loads
    2. Extract JSON from ```json ... ``` or ``` ... ``` fences
    3. Give up -> return None

    NOTE: A third fallback that greedily regex-matched any {...} block
    anywhere in raw_text used to exist here. It was removed because it
    could match an unrelated JSON-looking snippet embedded inside a
    markdown code example in the LLM's answer (e.g. a sample payload
    copied from retrieved context), silently discarding the real
    markdown answer and replacing it with junk from that snippet.
    """

    if not raw_text:
        return None

    try:
        return _coerce_schema(json.loads(raw_text))
    except (json.JSONDecodeError, TypeError):
        pass

    fence_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        raw_text,
        re.DOTALL
    )

    if fence_match:
        try:
            return _coerce_schema(json.loads(fence_match.group(1)))
        except (json.JSONDecodeError, TypeError):
            pass

    return None


@observe(name="analyze_input")
def traced_analyze_input(query):

    return analyze_input(
        query,
        deprecated_apis
    )


@observe(name="hybrid_retrieval")
def traced_retrieve(
    query,
    requested_version,
    python_mode,
    detected_apis
):

    with redirect_stdout(io.StringIO()):

        result = retrieve_documents(
            query=query,
            requested_version=requested_version,
            python_mode=python_mode,
            detected_apis=detected_apis,
            vector_store=vector_store,
            all_chunks=all_chunks
        )

    langfuse.update_current_span(
        metadata={
            "num_docs_retrieved": len(result["top_docs"]),
            "requested_version": requested_version
        }
    )

    return result


@observe(name="hallucination_guard")
def traced_guard(
    python_mode,
    requested_api,
    top_docs
):

    return hallucination_guard(
        python_mode=python_mode,
        requested_api=requested_api,
        top_docs=top_docs
    )


@observe(name="prompt_builder")
def traced_build_prompt(
    python_mode,
    error_mode,
    context,
    query,
    detected_apis
):

    return build_prompt(
        python_mode=python_mode,
        error_mode=error_mode,
        context=context,
        query=query,
        detected_apis=detected_apis
    )


@observe(name="llm_call", as_type="generation")
def traced_llm_call(prompt):

    response = llm.invoke(prompt)

    response_metadata = getattr(response, "response_metadata", {}) or {}

    usage = (
        getattr(response, "usage_metadata", None)
        or response_metadata.get("token_usage")
        or {}
    )

    input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
    output_tokens = usage.get("output_tokens") or usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")

    langfuse.update_current_generation(
        input=prompt,
        output=response.content,
        model="openai/gpt-oss-120b",
        usage_details={
            "input": input_tokens,
            "output": output_tokens,
            "total": total_tokens
        } if usage else None
    )

    return response


@observe(name="langchain_migration")
def run_pipeline(query: str):

    langfuse.update_current_span(input=query)

    analysis = traced_analyze_input(query)

    requested_version = analysis["version"]
    python_mode = analysis["python_mode"]
    error_mode = analysis["error_mode"]
    detected_apis = analysis["detected_apis"]
    requested_api = analysis["requested_api"]

    langfuse.update_current_span(
        metadata={
            "python_mode": python_mode,
            "error_mode": error_mode,
            "version": requested_version,
            "detected_apis": detected_apis
        }
    )

    result = traced_retrieve(
        query=query,
        requested_version=requested_version,
        python_mode=python_mode,
        detected_apis=detected_apis
    )

    top_docs = result["top_docs"]

    if not top_docs:

        msg = "No relevant documents found."

        langfuse.update_current_span(output=msg)
        langfuse.flush()

        no_docs_payload = {
            **REQUIRED_FIELDS,
            "summary": msg
        }

        return {
            "answer": no_docs_payload,
            **no_docs_payload,
            "contexts": [],
            "documents": []
        }

    context = result["context"]

    if not traced_guard(
        python_mode=python_mode,
        requested_api=requested_api,
        top_docs=top_docs
    ):

        msg = "The requested API is not mentioned in the retrieved context."

        langfuse.update_current_span(output=msg)
        langfuse.flush()

        guard_fail_payload = {
            **REQUIRED_FIELDS,
            "summary": msg
        }

        return {
            "answer": guard_fail_payload,
            **guard_fail_payload,
            "contexts": [doc.page_content for doc in top_docs],
            "documents": top_docs
        }

    prompt = traced_build_prompt(
        python_mode=python_mode,
        error_mode=error_mode,
        context=context,
        query=query,
        detected_apis=detected_apis
    )

    response = traced_llm_call(prompt)

    raw_text = response.content

    parsed = parse_llm_json(raw_text)

    langfuse.update_current_span(
        output=parsed if parsed is not None else raw_text
    )

    langfuse.flush()

    contexts = [doc.page_content for doc in top_docs]

    if parsed is not None:
        return {
            "answer": parsed,
            **parsed,
            "contexts": contexts,
            "documents": top_docs
        }

    # total parse failure -> still full schema, raw text goes in "summary"
    parse_fail_payload = {
        **REQUIRED_FIELDS,
        "summary": raw_text
    }

    return {
        "answer": parse_fail_payload,
        **parse_fail_payload,
        "contexts": contexts,
        "documents": top_docs
    }