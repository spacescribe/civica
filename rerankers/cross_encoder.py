from sentence_transformers import CrossEncoder

reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L2-v2")

def rerank(query: str, documents: list, top_n=5):
    """
        Rerank a list of documents based on the relevance to the query and return top_n results.

        Args:
            query: User query string
            documents: List of document objects returned by retriever
            top_n: Number of documents to return

        Returns:
            List of top_n documents sorted by relevance to the query
    """

    pairs = [(query, doc.page_content) for doc in documents]

    scores = reranker_model.predict(pairs)
    print("Reranker scores: ", scores)
    scored_docs = list(zip(scores, documents))
    scored_docs.sort(key= lambda x: x[0], reverse=True)

    reranked = [doc for _, doc in scored_docs[:top_n]]

    return reranked