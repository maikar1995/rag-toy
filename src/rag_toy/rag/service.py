

class RAGService:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    async def ask(self, query: str, context_id: str | None = None):
        results = self.retriever.search(query=query, top_k=5, filters=None)
        return self.generator.generate(query=query, search_results=results)