from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


def _ensure_metadata(doc):
    """
    Guarantee that every returned document carries the metadata fields
    the UI depends on: version, doc_type, section, source.
    Fills in 'Unknown' for anything missing so app.py never has to guess.
    """

    if not hasattr(doc, "metadata") or doc.metadata is None:
        doc.metadata = {}

    doc.metadata.setdefault("version", "Unknown")
    doc.metadata.setdefault("doc_type", "Unknown")
    doc.metadata.setdefault("section", "Unknown")

    if "source" not in doc.metadata:
        doc.metadata["source"] = (
            doc.metadata.get("source_file")
            or doc.metadata.get("file_path")
            or doc.metadata.get("file")
            or "Unknown"
        )

    return doc


def retrieve_documents(
    query,
    requested_version,
    python_mode,
    detected_apis,
    vector_store,
    all_chunks
):

    if detected_apis:
        retrieval_query = " ".join(detected_apis)
    else:
        retrieval_query = query

    print()
    print("Retrieval Query")
    print("-" * 40)
    print(retrieval_query)

    dense_docs = vector_store.max_marginal_relevance_search(
        retrieval_query,
        k=50,
        fetch_k=100
    )

    if requested_version:
        dense_docs = [
            doc
            for doc in dense_docs
            if doc.metadata.get("version") == requested_version
        ]

    dense_docs = dense_docs[:20]

    tokenized_corpus = [
        doc.page_content.split()
        for doc in all_chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    bm25_scores = bm25.get_scores(retrieval_query.split())

    ranked_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )

    bm25_docs = []

    for idx in ranked_indices:
        doc = all_chunks[idx]

        if requested_version:
            if doc.metadata.get("version") != requested_version:
                continue

        bm25_docs.append(doc)

        if len(bm25_docs) == 20:
            break

    merged = {}

    for doc in dense_docs:
        merged[doc.page_content] = doc

    for doc in bm25_docs:
        merged[doc.page_content] = doc

    candidate_docs = list(merged.values())

    print()
    print("Retrieval Statistics")
    print("-" * 40)
    print("Dense:", len(dense_docs))
    print("BM25:", len(bm25_docs))
    print("Merged:", len(candidate_docs))

    try:
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        pairs = [
            (retrieval_query, doc.page_content)
            for doc in candidate_docs
        ]

        rerank_scores = reranker.predict(pairs)

        ranked_docs = sorted(
            zip(rerank_scores, candidate_docs),
            key=lambda x: x[0],
            reverse=True
        )

        top_docs = [doc for score, doc in ranked_docs[:5]]

    except Exception as e:
        print()
        print("=" * 100)
        print("RERANKER WARNING")
        print("=" * 100)
        print()
        print(e)
        print()
        print("Falling back to Hybrid Retrieval without reranking...")

        top_docs = candidate_docs[:5]

    top_docs = [_ensure_metadata(doc) for doc in top_docs]

    print()
    print("TOP DOCUMENTS\n")

    for i, doc in enumerate(top_docs, start=1):
        print("=" * 100)
        print(f"Rank {i}")
        print(doc.metadata)
        print()
        print(doc.page_content[:700])
        print()

    context = "\n\n".join(
        doc.page_content
        for doc in top_docs
    )

    return {
        "retrieval_query": retrieval_query,
        "top_docs": top_docs,
        "context": context
    }


def hallucination_guard(
    python_mode,
    requested_api,
    top_docs
):

    if python_mode:
        return True

    if not requested_api:
        return True

    normalized_api = requested_api.replace("()", "").strip().lower()

    for doc in top_docs:
        if normalized_api in doc.page_content.lower():
            return True

    return False