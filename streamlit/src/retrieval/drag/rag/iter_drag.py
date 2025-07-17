"""
Iterative-DRAG Implementation
"""
from typing import List, Dict, Tuple
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


from retrieval.drag.rag import DRAG
from retrieval.drag.utils.data_types import (
    QueryResult,
    RAGExample,
    RAGConfig
)
from retrieval.vectorstore_setup import (
    retriever_with_score,
)


class IterDRAG(DRAG):
    """
    Implementation of Iterative Demonstration-based RAG (IterDRAG)
    as described in the paper for handling complex multi-hop queries
    """

    def __init__(self, document_store: List[Document], embedding_model: OpenAIEmbeddings, config: RAGConfig):
        super().__init__(document_store, embedding_model, config)
        self.cur_iteration = 0

    def _format_prompt(
        self, examples: List[RAGExample], documents: List[Document],
        query: str, sub_queries: List[str] = None, intermediate_answers: List[str] = None
    ) -> str:
        """
        Format prompt for iterative processing with examples and current state
        """
        prompt = (
            "You are an expert in breaking down and answering complex questions.\n"
            "Answer the following question using the provided context."
            "Your response MUST start exactly with one of these prefixes:\n"
            "  Follow up: <your sub-question>\n"
            "  Intermediate answer: <your interim answer>\n"
            "  So the final answer is: <your final answer>\n\n"
            "Then follow the Self-Ask format:\n"
            "1. If you need more information, begin with 'Follow up:'.\n"
            "2. When you have a partial answer, begin with 'Intermediate answer:'.\n"
            "3. When ready to conclude, begin with 'So the final answer is:'.\n"
        )

        # Add demonstrations
        prompt += "Context-1:\n"
        for ex in examples:
            for doc in ex.documents:
                prompt += f"{doc.page_content}\n"
            prompt += f"\nQuestion: {ex.query}\n"

            if ex.sub_queries and ex.intermediate_answers:
                for sq, ia in zip(ex.sub_queries, ex.intermediate_answers):
                    prompt += f"Follow up: {sq}\n"
                    prompt += f"Intermediate answer: {ia}\n"

            prompt += f"So the final answer is: {ex.answer}\n\n"

        # Add current query context
        prompt += "Context-2:\n"
        for doc in documents:
            prompt += f"{doc.page_content}\n\n"
        prompt += f"\nQuestion: {query}"

        # Add current progress if available
        if sub_queries and intermediate_answers:
            for sq, ia in zip(sub_queries, intermediate_answers):
                prompt += f"Follow up: {sq}\n"
                prompt += f"Intermediate answer: {ia}\n"
        return prompt

    def _format_final_prompt(
        self, examples: List[RAGExample], documents: List[Document],
        query: str, sub_queries: List[str] = None, intermediate_answers: List[str] = None
    ) -> str:
        """
        Format prompt for iterative processing with examples and current state
        """
        prompt = (
            "You are a QA expert skilled at extracting and summarizing information from provided context.\n"
            "Carefully review the context below, then answer the question in clear, concise bullet points.\n"
            "Begin your response exactly with this line (including the colon):\n"
            "So the final answer is:\n\n"
        )

        # Add demonstrations
        prompt += "Context-1:\n"
        for ex in examples:
            for doc in ex.documents:
                prompt += f"{doc.page_content}\n"
            prompt += f"Example Question: {ex.query}\n"

            if ex.sub_queries and ex.intermediate_answers:
                for sq, ia in zip(ex.sub_queries, ex.intermediate_answers):
                    prompt += f"Follow up: {sq}\n"
                    prompt += f"Intermediate answer: {ia}\n"

            prompt += f"So the final answer is: {ex.answer}\n\n"

        # Add current query context
        prompt += "Context-2:\n"
        for doc in documents:
            prompt += f"{doc.page_content}\n"
        prompt += f"\nQuestion: {query}"

        # Add current progress if available
        if sub_queries and intermediate_answers:
            for sq, ia in zip(sub_queries, intermediate_answers):
                prompt += f"Follow up: {sq}\n"
                prompt += f"Intermediate answer: {ia}\n"
        return prompt

    def _generate_next_step(self, prompt: str) -> Tuple[str, str]:
        """
        Generate the next step in the iterative process:
        - sub-query for additional information
        - intermediate answer
        - final answer
        """
        response = self.llm.invoke(prompt).content[0]['text']

        if "So the final answer is:" in response:
            # Extract intermediate answer
            final_answer = response.split("So the final answer is:")[1].strip()
            return "final", final_answer
        elif "Follow up:" in response:
            # Extract sub-query
            sub_query = response.split("Follow up:")[1].split("\n")[0].strip()
            return "sub_query", sub_query
        elif "Intermediate answer:" in response:
            # Extract final answer
            iter_answer = response.split("Intermediate answer:")[1].split("\n")[0].strip()
            return "intermediate", iter_answer

    def _retrieve_for_sub_query(self, sub_query: str, existing_docs: List[Document]) -> List[Document]:
        """
        Retrieve additional documents for a sub-query while avoiding duplicates
        """
        # Retrieve new documents
        new_docs = retriever_with_score.invoke({
            "vectorstore": self.document_store,
            "query": sub_query,
            "k": self.config.num_documents
        })

        # Filter out duplicates
        existing_ids = set(doc.metadata["id"] for doc in existing_docs)
        filtered_docs = [doc for doc in new_docs if doc.metadata["id"] not in existing_ids]

        return filtered_docs

    def process_query(self, query: str, examples: List[RAGExample]) -> QueryResult:
        """
        Process a query using IterDRAG approach:
        1. Start with initial retrieval
        2. Iteratively:
            - Generate sub-queries
            - Retrieve additional context
            - Generate intermediate answers
            - Produce final answer
        """
        # Initial retrieval
        documents = retriever_with_score.invoke({
            "vectorstore": self.document_store,
            "query": query,
            "k": self.config.num_documents
        })

        sub_queries = []
        intermediate_answers = []
        total_context_length = 0

        # Keep track of highest relevance score
        max_confidence = max(doc.metadata["score"] for doc in documents)

        # Iterative processing
        for iteration in (range(self.config.max_iterations)):
            # Format current state as prompt
            prompt = self._format_prompt(
                examples[:self.config.num_shots],
                documents,
                query,
                sub_queries,
                intermediate_answers
            )

            # Update context length tracking
            step_length = len(self.embedding_model.embed_query(prompt))
            total_context_length += step_length

            # Check if we've exceeded max context length
            if total_context_length > self.config.max_context_length:
                # Force final answer generation
                final_prompt = self._format_final_prompt(
                    examples[:1],  # Use fewer examples to fit context
                    documents[-10:],  # Use most recent documents
                    query,
                    sub_queries,
                    intermediate_answers
                )
                _, final_answer = self._generate_next_step(final_prompt)
                break

            # Generate next step
            step_type, content = self._generate_next_step(prompt)

            if step_type == "final":
                final_prompt = prompt
                final_answer = content
                break

            elif step_type == "sub_query":
                sub_queries.append(content)

                # Retrieve additional documents for sub-query
                new_docs = self._retrieve_for_sub_query(content, documents)
                documents.extend(new_docs)

                # Update confidence if we found more relevant documents
                if new_docs:
                    max_confidence = max(
                        max_confidence,
                        max(doc.metadata["score"] for doc in new_docs)
                    )

            else:  # intermediate answer
                intermediate_answers.append(content)
            print(f"{'=' * 50}\n\nIteration {iteration + 1}\nPrompt:\n{prompt}\nAnswer:\n{content}\n")

            # Force final answer on last iteration
            if iteration == self.config.max_iterations - 1:
                print("Final iteration reached, generating final answer...")
                # Force final answer generation
                final_prompt = self._format_final_prompt(
                    examples[:1],  # Use fewer examples to fit context
                    documents[-10:],  # Use most recent documents
                    query,
                    sub_queries,
                    intermediate_answers
                )
                _, final_answer = self._generate_next_step(final_prompt)
                print(f"{'=' * 50}\nFinal prompt:\n{final_prompt}\n{'=' * 50}\nFinal answer:\n{final_answer}\n")

            self.cur_iteration += 1

        with open("./process/rag/output/prompt_iter_drag.txt", "w") as f:
            f.write(final_prompt)

        with open("./process/rag/output/answer_iter_drag.txt", "w") as f:
            f.write(final_answer)

        return QueryResult(
            query=query,
            documents=documents,
            answer=final_answer,
            confidence=max_confidence,
            sub_queries=sub_queries,
            intermediate_answers=intermediate_answers,
            effective_context_length=total_context_length
        )

    def evaluate_decomposition(self, query_result: QueryResult, ground_truth: str) -> Dict[str, float]:
        """
        Evaluate the quality of query decomposition and intermediate steps
        """
        metrics = {
            'num_steps': len(query_result.sub_queries),
            'avg_docs_per_step': len(query_result.documents) / max(1, len(query_result.sub_queries)),
            'context_efficiency': query_result.confidence / query_result.effective_context_length
        }

        return metrics
