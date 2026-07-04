import os
import re
import json
import pickle

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

DATA_FOLDER = "data/cleaned"


def is_valid_api(api):

    api = api.strip()

    if len(api) < 3:
        return False

    if "\n" in api:
        return False

    if " " in api:
        return False

    if api.startswith(("0.", "1.", "2.", "^")):
        return False

    if api.endswith((".md", ".toml")):
        return False

    if "=" in api:
        return False

    if ":" in api:
        return False

    ignore = {
        "The",
        "README",
        "RESUME",
        "Makefile",
        "Documents",
        "Optional",
        "Schema",
        "async",
        "beta",
        "minimal",
        "data",
        "model",
        "runtime",
        "type",
        "values",
        "tools",
        "retrievers",
        "langchain",
        "langgraph",
        "pydantic"
    }

    if api in ignore:
        return False

    if "." in api:
        return True

    if api.endswith("()"):
        return True

    if "_" in api:
        return True

    if re.match(r"^[A-Z][A-Za-z0-9_]+$", api):
        return True

    if re.match(r"^[a-z_][A-Za-z0-9_]+$", api):
        return True

    return False


docs = []

for root, dirs, files in os.walk(DATA_FOLDER):

    for file in files:

        if not file.endswith((".md", ".mdx")):
            continue

        path = os.path.join(root, file)

        loader = TextLoader(
            path,
            encoding="utf-8"
        )

        loaded_docs = loader.load()

        parts = path.replace("\\", "/").split("/")

        version = parts[2]

        doc_type = "release_notes"

        if "migration_guides" in parts:
            doc_type = "migration_guides"

        section = ""

        if "migration_guides" in parts:

            guide_index = parts.index("migration_guides")

            if guide_index + 1 < len(parts) - 1:
                section = parts[guide_index + 1]

        for doc in loaded_docs:

            doc.metadata["version"] = version
            doc.metadata["doc_type"] = doc_type
            doc.metadata["section"] = section
            doc.metadata["file_name"] = file
            doc.metadata["source"] = path

            docs.append(doc)

print("Documents Loaded:", len(docs))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
    separators=[
        "\n## ",
        "\n### ",
        "\n#### ",
        "\n",
        " ",
        ""
    ]
)

chunks = splitter.split_documents(docs)

print("Chunks Created:", len(chunks))

with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Chunks Saved: chunks.pkl")

deprecated_apis = set()

for chunk in chunks:

    matches = re.findall(
        r"`([^`]+)`",
        chunk.page_content
    )

    for match in matches:

        if is_valid_api(match):

            deprecated_apis.add(
                match.strip()
            )

deprecated_apis = sorted(deprecated_apis)

with open(
    "deprecated_apis.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        deprecated_apis,
        f,
        indent=4
    )

print(
    "Deprecated APIs Saved:",
    len(deprecated_apis)
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    path="./qdrant_db",
    collection_name="langchain_migration_docs"
)

print("Qdrant Vector DB Created")