"""
DRAG Implementation
"""

# TODO: Implement DRAG functionality
from typing import List, Optional
from ..utils.data_types import QueryResult, RAGExample, RAGConfig
from ..utils.helper_functions import ChatOpenAI, OpenAIEmbeddings, Document, retriever_with_score


class DRAG:
    """Implementation of Demonstration-based RAG"""
    def __init__(
        self,
        document_store: List[Document],
        embedding_model: OpenAIEmbeddings,
        config: RAGConfig
    ):
        self.document_store = document_store
        self.embedding_model = embedding_model
        self.config = config
        self.llm = ChatOpenAI(**config.model_settings)

    def _format_prompt(
        self,
        examples: List[RAGExample],
        documents: List[Document],
        query: str
    ) -> str:
        """
        Format prompt with demonstrations and query
        """
        prompt = (
            "You are a QA expert skilled at extracting and summarizing information from provided context.\n"
            "Carefully review the context below, then answer the question in clear, concise bullet points.\n"
        )

        # Add demonstrations
        prompt += "Context-1:\n"
        for ex in examples:
            for doc in ex.documents:
                prompt += f"{doc.page_content}\n"
            prompt += f"Question: {ex.query}\nAnswer: {ex.answer}\n\n"

        # Add test query
        prompt += "Context-2:\n"
        for doc in documents:
            prompt += f"{doc.page_content}\n"
        prompt += f"\nQuestion: {query}\nAnswer:"

        return prompt

    def process_query(
        self,
        query: str,
        examples: List[RAGExample]
    ) -> QueryResult:
        """
        Process query using DRAG
        """
        # Retrieve relevant documents
        documents = retriever_with_score.invoke({
            "vectorstore": self.document_store,
            "query": query,
            "k": self.config.num_documents
        })

        # Format prompt
        prompt = self._format_prompt(examples[:self.config.num_shots], documents, query)

        # Generate answer
        answer = self.llm.invoke(prompt).content[0]['text']

        with open("./process/rag/output/prompt_drag.txt", "w") as f:
            f.write(prompt)
        with open("./process/rag/output/answer_drag.txt", "w") as f:
            f.write(answer)

        # Calculate effective context length
        effective_length = len(self.embedding_model.embed_query(prompt))

        return QueryResult(
            query=query,
            documents=documents,
            answer=answer,
            confidence=max(doc.metadata["score"] for doc in documents),
            effective_context_length=effective_length
        )
