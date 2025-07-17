# Functions to enhance the retriever

import os
import sys
import numpy as np

from scipy.special import logsumexp

if os.path.abspath(os.path.join(os.path.dirname('__file__'), '..', '..')) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname('__file__'), '..', '..')))
# print(sys.path)
# helper functions
import helper_functions as utils


########################################
#  query functions
########################################
class QueryTransformation():
    """
    Query Transformation Functions
    This class contains methods to transform the original query into different forms to improve retrieval.
    """
    def __init__(self, originalQuery, modelSetting):
        """
        Args:
        originalQuery (str): The original user query
        modelSetting (**kwargs): The setting of model to use for generating the step-back query.
        """
        self.originalQuery = originalQuery
        self.modelSetting = modelSetting

    def generate_rewrite_query(self):
        """
        Rewrite the original query to improve retrieval.

        Returns:
        str: The rewritten query
        """
        # print(f"model setting: {modelSetting}")

        rewrite_llm = utils.ChatOpenAI(**self.modelSetting)
        queryRewrite_template = """You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system.
            Given the original query, please rewrite it to be more specific, detailed, and likely to retrieve relevant information.

            Original query: {original_query}

            Rewritten query:"""
        queryRewrite_prompt = utils.PromptTemplate(
            input_variables=["original_query"],
            template=queryRewrite_template,
        )

        # Create an LLMChain for query rewriting
        queryRewriter = queryRewrite_prompt | rewrite_llm
        response = queryRewriter.invoke(self.originalQuery)

        return response

    def generate_stepBack_query(self):
        """
        Generate a step-back query to retrieve broader context.

        Returns:
        str: The step-back query
        """
        stepBack_llm = utils.ChatOpenAI(**self.modelSetting)
        stepBack_template = """You are an AI assistant tasked with generating broader, more general queries to improve context retrieval in a RAG system.
            Given the original query, please generate a step-back query that is more general and can help retrieve relevant background information.

            Original query: {original_query}

            Step-back query:"""
        stepBack_prompt = utils.PromptTemplate(
            input_variables=["original_query"],
            template=stepBack_template,
        )

        # Create an LLMChain for step-back prompting
        stepBack_chain = stepBack_prompt | stepBack_llm
        response = stepBack_chain.invoke(self.originalQuery)
        return response

    def generate_subqueryDecomposition_query(self):
        """
        Decompose the original query into simpler sub-queries.

        Returns:
        List[str]: A list of simpler sub-queries
        """
        subqueryDecomposition_llm = utils.ChatOpenAI(**self.modelSetting)
        subqueryDecomposition_template = """You are an AI assistant tasked with breaking down complex queries into simpler sub-queries for a RAG system.
            Given the original query, please decompose it into 5 simpler sub-queries that, when answered together, would provide a comprehensive response to the original query.

            Original query: {original_query}

            example: What are the impacts of climate change on the environment?

            Sub-queries:
            What are the impacts of climate change on biodiversity?
            How does climate change affect the oceans?
            What are the effects of climate change on agriculture?
            What are the impacts of climate change on human health?
            How does climate change influence weather patterns?

            Sub-queries:"""
        subqueryDecomposition_prompt = utils.PromptTemplate(
            input_variables=["original_query"],
            template=subqueryDecomposition_template,
        )

        # Create an LLMChain for sub-query decomposition
        subqueryDecomposition_chain = subqueryDecomposition_prompt | subqueryDecomposition_llm
        response = subqueryDecomposition_chain.invoke(self.originalQuery)
        return response


########################################
#  embedding functions
########################################
class HyDE():
    """
    Hypothetical Document Embeddings

    Uses the LLM to generate a hypothetical document that answers the user's query.
    This document is then used to retrieve relevant chunks from the vector store.
    """
    def __ini__(self, modelSetting, chunkSize=500):
        # self.llm = ChatOpenAI(temperature=0, model_name="gpt-4.1-nano", max_tokens=4000)
        self.llm = utils.ChatOpenAI(**modelSetting)
        self.chunk_size = chunkSize
        self.hyde_prompt = utils.PromptTemplate(
            input_variables=["query", "chunk_size"],
            template="""Given the question '{query}', generate a hypothetical document that directly answers this question. The document should be detailed and in-depth.
            the document size has be exactly {chunk_size} characters.""",
        )
        self.hyde_chain = self.hyde_prompt | self.llm

        self.hypothetical_results = None
        self.query = None

    def generate_hypothetical_document(self, query):
        self.query = query
        input_variables = {"query": query, "chunk_size": self.chunk_size}
        self.hypothetical_results = self.hyde_chain.invoke(input_variables)
        return self.hypothetical_results

    def retrieve_vectorstore(self, k=3):
        hypothetical_doc = self.hypothetical_results.content
        similar_docs = self.vectorstore.similarity_search(hypothetical_doc, k=k)
        return similar_docs, hypothetical_doc


class HyPE():
    """
    Hypothetical Prompt Embeddings
    Uses the LLM to generate multiple hypothetical questions for a single chunk.
    These questions will be used as 'proxies' for the chunk during retrieval.
    """
    def __init__(self, modelSetting):
        self.llm = utils.ChatOpenAI(**modelSetting)
        self.embedding = utils.OpenAIEmbeddings(model="text-embedding-3-small")

        self.question_gen_prompt = utils.PromptTemplate(
            input_variables=["chunk_text"],
            template="""Analyze the input text and generate essential questions that, when answered, capture the main points and core meaning of the text.
            The questions should be exhaustive and understandable without context.
            When possible, named entities should be referenced by their full name.
            Only answer with questions where each question should be written in its own line (separated by newline) with no prefix.

            Text:
            {chunk_text}

            Questions:
            """
        )
        self.question_chain = self.question_gen_prompt | self.llm

        self.chunk_text = None

    def generate_hypothetical_embeddings(self, chunk_text):
        input_variables = {"chunk_text": chunk_text}
        hypothetical_results = self.question_chain.invoke(input_variables)
        questions = hypothetical_results.content.replace("\n\n", "\n").split("\n")
        question_embeddings = self.embedding.embed_documents(questions)
        return chunk_text, hypothetical_results, questions, question_embeddings

    def load_to_vectorstore_parallel(self, chunks, collectionName, pathPersist):
        """
        Creates and populates a FAISS vector store from a list of text chunks.

        This function processes a list of text chunks in parallel, generating
        hypothetical prompt embeddings for each chunk.
        The embeddings are stored in a FAISS index for efficient similarity search.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        vectorstore = None

        with ThreadPoolExecutor() as pool:
            # Use threading to speed up generation of prompt embeddings
            futures = [pool.submit(self.generate_hypothetical_embeddings, c) for c in chunks]

            for f in utils.tqdm(as_completed(futures), total=len(chunks), ncols=100):
                chunk, response, questions, vectors = f.result()

                vectorstore = utils.get_vectorstore(
                    collectionName=collectionName,
                    path=pathPersist,
                    database=utils.VectorDatabase.FAISS,
                )

                # Pair the chunk's content with each generated embedding vector
                # Each chunk is inserted multiple times, once for each prompt vector
                chunks_with_embedding_vectors = [(chunk.page_content, vec) for vec in vectors]

                # Add embeddings to the store
                vectorstore.add_embeddings(chunks_with_embedding_vectors)

        return vectorstore


########################################
#  chunking functions
########################################
class RSE():
    """
    Relevant Segment Extraction (RSE)
    Dynamically constructing multi-chunk segments of text that are relevant to a given query.
    """
    def __init__(self):
        pass

    def transform(x):
        """
        Transformation function to map the absolute relevance value to a value that is more uniformly distributed between 0 and 1. The relevance values given by the Cohere reranker tend to be very close to 0 or 1. This beta function used here helps to spread out the values more uniformly.

        Args:
            x (float): The absolute relevance value returned by the Cohere reranker

        Returns:
            float: The transformed relevance value
        """
        from scipy.stats import beta
        a, b = 0.4, 0.4  # These can be adjusted to change the distribution shape
        return beta.cdf(x, a, b)

    def plot_relevance_scores(self, chunkValues: utils.List[float], startIdx: int = None, endIdx: int = None):
        """
        Visualize the relevance scores of each chunk in the document to the search query

        Args:
            chunkValues (list): List of relevance values for each chunk
            startIndex (int): Start index of the chunks to be plotted
            endIndex (int): End index of the chunks to be plotted

        Returns:
            None

        Plots:
            Scatter plot of the relevance scores of each chunk in the document to the search query
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 5))
        plt.title("Similarity of each chunk in the document to the search query")
        plt.ylim(0, 1)
        plt.xlabel("Chunk index")
        plt.ylabel("Query-chunk similarity")
        if startIdx is None:
            startIdx = 0
        if endIdx is None:
            endIdx = len(chunkValues)
        plt.scatter(range(startIdx, endIdx), chunkValues[startIdx: endIdx])

    def get_best_segments(relevanceValues: list, maxLength: int, overallMaxLength: int, minimumValue: float):
        """
        This function takes the chunk relevance values and then runs an optimization algorithm to find the best segments. In more technical terms, it solves a constrained version of the maximum sum subarray problem.

        Note: this is a simplified implementation intended for demonstration purposes. A more sophisticated implementation would be needed for production use and is available in the dsRAG library.

        Args:
            relevanceValues (list): a list of relevance values for each chunk of a document
            maxLength (int): the maximum length of a single segment (measured in number of chunks)
            overallMaxLength (int): the maximum length of all segments (measured in number of chunks)
            minimumValue (float): the minimum value that a segment must have to be considered

        Returns:
            best_segments (list): a list of tuples (start, end) that represent the indices of the best segments (the end index is non-inclusive) in the document
            scores (list): a list of the scores for each of the best segments
        """
        best_segments = []
        scores = []
        totalLength = 0
        while totalLength < overallMaxLength:
            # Find the best remaining segment
            best_segment = None
            best_value = 0
            for start in range(len(relevanceValues)):
                # Skip over negative value starting points
                if relevanceValues[start] < 0:
                    continue
                for end in range(start + 1, min(start + maxLength + 1, len(relevanceValues) + 1)):
                    # Skip over negative value ending points
                    if relevanceValues[end - 1] < 0:
                        continue
                    # Check if this segment overlaps with any of the best segments and skip if it does
                    if any(start < seg_end and end > seg_start for seg_start, seg_end in best_segments):
                        continue
                    # Check if this segment would push us over the overall max length and skip if it does
                    if totalLength + end - start > overallMaxLength:
                        continue

                    # Define segment value as the sum of the relevance values of its chunks
                    segmentValue = sum(relevanceValues[start:end])
                    if segmentValue > best_value:
                        best_value = segmentValue
                        best_segment = (start, end)

            # If we didn't find a valid segment the we're done
            if best_segment is None or best_value < minimumValue:
                break

            # Otherwise, add the segment to the list of best segments
            best_segments.append(best_segment)
            scores.append(best_value)
            totalLength += best_segment[1] - best_segment[0]

        return best_segments, scores


class SemanticChunking():
    """
    Dividing documents based on semantic coherence rather than fixed sizes.
    """
    def __init__(self, embeddings, breakpointType, breakpointThreshold):
        """
        Breakpoint types:
        'percentile': all differences between sentences are calculated, and then any difference greater than the X percentile is split.
        'standard_deviation': any difference greater than X standard deviations is split.
        'interquartile': the interquartile distance is used to split chunks.
        """
        from langchain_experimental.text_splitter import SemanticChunker
        self.breakpoint_type = breakpointType
        self.breakpoint_threshold = breakpointThreshold
        self.text_splitter = SemanticChunker(
            embeddings,
            breakpoint_type=self.breakpoint_type,
            breakpoint_threshold=self.breakpoint_threshold,
        )

    def split_into_chunks(self, content: str):
        """
        Args:
            content (str): The text content to be chunked.

        Returns:
            list[Document]: A list of Document objects, each containing a chunk of text.
        """
        return self.text_splitter.create_documents([content])


########################################
#  retrieval/ranking functions
########################################
class FusionRetriever():
    """
    Fusion Retrieval
    Combine keyword-based (BM25) search with vector-based (similarity) search for more comprehensive and accurate retrieval.
    """
    def __init__(self):
        pass

    def create_bm25_index(documents: utils.List[utils.Document]) -> utils.BM25Okapi:
        """
        Create a BM25 index from the given documents.

        BM25 (Best Matching 25) is a ranking function used in information retrieval.
        It's based on the probabilistic retrieval framework and is an improvement over TF-IDF.

        Args:
        documents (List[Document]): List of documents to index.

        Returns:
        BM25Okapi: An index that can be used for BM25 scoring.
        """
        # Tokenize each document by splitting on whitespace
        # This is a simple approach and could be improved with more sophisticated tokenization
        tokenized_docs = [doc.page_content.split() for doc in documents]
        return utils.BM25Okapi(tokenized_docs)

    def fusion_retrieval(vectorstore, bm25, query: str, k: int = 5, alpha: float = 0.5) -> utils.List[utils.Document]:
        """
        Perform fusion retrieval combining keyword-based (BM25) and vector-based search.

        Args:
        vectorstore (VectorStore): The vectorstore containing the documents.
        bm25 (BM25Okapi): Pre-computed BM25 index.
        query (str): The query string.
        k (int): The number of documents to retrieve.
        alpha (float): The weight for vector search scores (1-alpha will be the weight for BM25 scores).

        Returns:
        List[Document]: The top k documents based on the combined scores.
        """

        epsilon = 1e-8

        # Step 1: Get all documents from the vectorstore
        all_docs = vectorstore.similarity_search("", k=vectorstore.index.ntotal)

        # Step 2: Perform BM25 search
        bm25_scores = bm25.get_scores(query.split())

        # Step 3: Perform vector search
        vector_results = vectorstore.similarity_search_with_score(query, k=len(all_docs))

        # Step 4: Normalize scores
        vector_scores = np.array([score for _, score in vector_results])
        vector_scores = 1 - (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores) + epsilon)

        bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + epsilon)

        # Step 5: Combine scores
        combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores

        # Step 6: Rank documents
        sorted_indices = np.argsort(combined_scores)[::-1]

        # Step 7: Return top k documents
        return [all_docs[i] for i in sorted_indices[:k]]


class HierarchicalIndexRetriever():
    """
    Hierarchical Indexing
    Create a hierarchical index of documents based on their content and structure.
    This allows for more efficient retrieval of relevant documents.
    """
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    async def encode_pdf_hierarchical(self, path, chunk_size=1000, chunk_overlap=200, is_string=False):
        """
        Asynchronously encodes a PDF book into a hierarchical vector store using OpenAI embeddings.
        Includes rate limit handling with exponential backoff.

        Args:
            path: The path to the PDF file.
            chunk_size: The desired size of each text chunk.
            chunk_overlap: The amount of overlap between consecutive chunks.

        Returns:
            A tuple containing two FAISS vector stores:
            1. Document-level summaries
            2. Detailed chunks
        """
        # Load PDF documents
        if not is_string:
            loader = utils.PyPDFLoader(path)
            documents = await utils.asyncio.to_thread(loader.load)
        else:
            text_splitter = utils.RecursiveCharacterTextSplitter(
                # Set a really small chunk size, just to show
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator=False,
            )
            documents = text_splitter.split_text(path)

        # Create document-level summaries
        summary_llm = utils.ChatOpenAI(temperature=0, model_name="gpt-4.1-nano", max_tokens=4000)
        summary_chain = utils.load_summarize_chain(summary_llm, chain_type="map_reduce")

        async def summarize_doc(doc):
            """
            Summarizes a single document with rate limit handling.

            Args:
                doc: The document to be summarized.

            Returns:
                A summarized Document object.
            """
            # Retry the summarization with exponential backoff
            summary_output = await self.retry_with_exponential_backoff(summary_chain.ainvoke([doc]))
            summary = summary_output['output_text']
            return utils.Document(page_content=summary, metadata={"source": path, "page": doc.metadata["page"], "summary": True})

        # Process documents in smaller batches to avoid rate limits
        batch_size = 5  # Adjust this based on your needs
        summaries = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i: i + batch_size]
            batch_summaries = await utils.asyncio.gather(*(summarize_doc(doc) for doc in batch))
            summaries.extend(batch_summaries)
            await utils.asyncio.sleep(1)

        # Split documents into detailed chunks
        text_splitter = utils.RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )
        detailed_chunks = await utils.asyncio.to_thread(text_splitter.split_documents, documents)

        # Update metadata for detailed chunks
        for i, chunk in enumerate(detailed_chunks):
            chunk.metadata.update({
                "chunk_id": i,
                "summary": False,
                "page": int(chunk.metadata.get("page", 0))
            })
        # Create embeddings
        embeddings = utils.OpenAIEmbeddings(model="text-embedding-3-small")

        # Create vector stores asynchronously with rate limit handling
        async def create_vectorstore(docs):
            """
            Creates a vector store from a list of documents with rate limit handling.

            Args:
                docs: The list of documents to be embedded.

            Returns:
                A FAISS vector store containing the embedded documents.
            """
            return await self.retry_with_exponential_backoff(
                utils.asyncio.to_thread(utils.FAISS.from_documents, docs, embeddings)
            )

        # Generate vector stores for summaries and detailed chunks concurrently
        summary_vectorstore, detailed_vectorstore = await utils.asyncio.gather(
            create_vectorstore(summaries),
            create_vectorstore(detailed_chunks)
        )

        return summary_vectorstore, detailed_vectorstore

    async def create_and_save_hierarchical_vectorstore(self, path_document, path_summary, path_detailed, summary_vectorstore, detailed_vectorstore):
        """
        Store the hierarchical index in the specified path.

        Args:
            path (str): The path to store the vector stores.
            summary_vectorstore (FAISS): The vector store for document-level summaries.
            detailed_vectorstore (FAISS): The vector store for detailed chunks.

        Returns:
            None
        """
        if os.path.exists(path_summary) and os.path.exists(path_detailed):
            embeddings = utils.OpenAIEmbeddings()
            summary_store = utils.FAISS.load_local(path_summary, embeddings, allow_dangerous_deserialization=True)
            detailed_store = utils.FAISS.load_local(path_detailed, embeddings, allow_dangerous_deserialization=True)
        else:
            summary_store, detailed_store = await self.encode_pdf_hierarchical(path_document)
            summary_store.save_local(path_summary)
            detailed_store.save_local(path_detailed)

    def retrieve_hierarchical(self, query, summary_vectorstore, detailed_vectorstore, k_summaries=3, k_chunks=5):
        """
        Performs a hierarchical retrieval using the query.

        Args:
            query: The search query.
            summary_vectorstore: The vector store containing document summaries.
            detailed_vectorstore: The vector store containing detailed chunks.
            k_summaries: The number of top summaries to retrieve.
            k_chunks: The number of detailed chunks to retrieve per summary.

        Returns:
            A list of relevant detailed chunks.
        """
        # Retrieve top summaries
        top_summaries = summary_vectorstore.similarity_search(query, k=k_summaries)

        relevant_chunks = []
        for summary in top_summaries:
            # For each summary, retrieve relevant detailed chunks
            page_number = summary.metadata["page"]
            page_filter = lambda metadata: metadata["page"] == page_number
            page_chunks = detailed_vectorstore.similarity_search(query, k=k_chunks, filter=page_filter)
            relevant_chunks.extend(page_chunks)

        return relevant_chunks


class DartboardRetriever():
    """Combine both relevance and diversity into a single scoring function and directly optimize for it."""
    def __init__(self, vectorstore, diversity_weight=1.0, relevance_weight=1.0, sigma=0.1):
        self.vectorstore = vectorstore
        self.diversity_weight = diversity_weight  # Weight for diversity in document selection
        self.relevance_weight = relevance_weight  # # Weight for relevance to query
        self.sigma = sigma  # Smoothing parameter for probability distribution

    def lognorm(self, dist: np.ndarray, sigma: float):
        """
        Calculate the log-normal probability for a given distance and sigma.
        """
        if sigma < 1e-9:
            return -np.inf * dist
        return -np.log(sigma) - 0.5 * np.log(2 * np.pi) - dist**2 / (2 * sigma**2)

    def greedy_dartsearch(
        self,
        query_distances: np.ndarray,
        document_distance: np.ndarray,
        documents: utils.List[str],
        num_results: int
    ) -> utils.Tuple[utils.List[str], utils.List[float]]:
        """
        Perform greedy dartboard search to select top k documents balancing relevance and diversity.

        Args:
            query_distances: Distance between query and each document
            document_distances: Pairwise distances between documents
            documents: List of document texts
            num_results: Number of documents to return

        Returns:
            Tuple containing:
            - List of selected document texts
            - List of selection scores for each document
        """
        # Avoid division by zero in probability calculation
        sigma = max(self.sigma, 1e-5)

        # Convert distances to probability distributions
        query_probabilities = self.lognorm(query_distances, sigma)
        document_probabilities = self.lognorm(document_distance, sigma)

        # Initialize with most relevant document
        most_relevant_idx = np.argmax(query_probabilities)
        selected_indices = np.array([most_relevant_idx])
        selection_scores = [1.0]  # dummy score for the first document
        # Get initial distances from the first selected document
        max_distance = document_probabilities[most_relevant_idx]

        # Select remaining documents
        while len(selected_indices) < num_results:
            # Update maximum distances considering new document
            updated_distances = np.maximum(max_distance, document_probabilities)

            # Calculate combined diversity and relevance scores
            combined_scores = (
                updated_distances * self.diversity_weight +
                query_probabilities * self.relevance_weight
            )

            # Normalize scores and mask already selected documents
            normalized_scores = logsumexp(combined_scores, axis=0)
            normalized_scores[selected_indices] = -np.inf

            # Select best remaining document
            best_idx = np.argmax(normalized_scores)
            best_score = np.max(normalized_scores)

            # Update tracking variables
            max_distance = updated_distances[best_idx]
            selected_indices = np.append(selected_indices, best_idx)
            selection_scores.append(best_score)

        # Return selected documents and their scores
        selected_documents = [documents[i] for i in selected_indices]
        return selected_documents, selection_scores

    def idx_to_text(self, idx: int):
        docstore_id = self.vectorstore.index_to_docstore_id[idx]
        document = self.vectorstore.docstore.search(docstore_id)
        return document.page_content

    def get_context(
        self,
        query: str,
        num_results: int = 5,
        oversampling_factor: int = 3,
    ) -> utils.Tuple[utils.List[str], utils.List[float]]:
        """
        Retrieve most relevant and diverse context items for a query using the dartboard algorithm.

        Args:
            query: The search query string
            num_results: Number of context items to return (default: 5)
            oversampling_factor: Factor to oversample initial results for better diversity (default: 3)

        Returns:
            Tuple containing:
            - List of selected context texts
            - List of selection scores

        Note:
            The function uses cosine similarity converted to distance. Initial retrieval
            fetches oversampling_factor * num_results items to ensure sufficient diversity
            in the final selection.
        """
        # Embed query and retrieve initial candidates
        query_embedding = self.vectorstore.embedding_function.embed_documents([query])
        _, candidate_indices = self.vectorstore.index.search(
            np.array(query_embedding),
            k=num_results * oversampling_factor
        )

        # Get document vectors and texts for candidates
        candidate_vectors = np.array(
            self.vectorstore.index.reconstruct_batch(candidate_indices[0])
        )
        candidate_texts = [self.idx_to_text(idx) for idx in candidate_indices[0]]

        # Calculate distance matrices
        # Using 1 - cosine_similarity as distance metric
        document_distances = 1 - np.dot(candidate_vectors, candidate_vectors.T)
        query_distances = 1 - np.dot(query_embedding, candidate_vectors.T)

        # Apply dartboard selection algorithm
        selected_texts, selection_scores = self.greedy_dartsearch(
            query_distances,
            document_distances,
            candidate_texts,
            num_results
        )

        return selected_texts, selection_scores


class FeedbackLoopRetriever():
    """
    Collect and utilize user feedback on the relevance and quality of retrieved documents and
    generated responses to fine-tune retrieval and ranking models.
    """
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def create_qu_chain(self, query):
        """
        Answer the query based on the retrieved documents.
        """
        retriever = self.vectorstore.as_retriever()
        llm = utils.ChatOpenAI(temperature=0, model_name="gpt-4.1-nano", max_tokens=4000)
        qa_chain = utils.RetrievalQA.from_chain_type(llm, retriever=retriever)
        return qa_chain

    def format_user_feedback(query, response, relevance, quality, comments=""):
        """format user feedback in a dictionary"""
        return {
            "query": query,
            "response": response,
            "relevance": int(relevance),
            "quality": int(quality),
            "comments": comments
        }

    def store_feedback(feedback, path):
        with open(f"{path}/feedback.jsonl", "a") as f:
            utils.json.dump(feedback, f)
            f.write("\n")

    def load_feedback_data(path):
        feedback_data = []
        try:
            with open(f"{path}/feedback.jsonl", "r") as f:
                for line in f:
                    feedback_data.append(utils.json.loads(line.strip()))
        except FileNotFoundError:
            print("No feedback data file found. Starting with empty feedback.")
        return feedback_data

    def adjust_relevance_scores(
        self,
        query: str,
        docs: utils.List[utils.Any],
        feedback_data: utils.List[utils.Dict[str, utils.Any]]
    ) -> utils.List[utils.Any]:
        """
        adjust files relevancy based on the feedbacks file
        """
        class Response(utils.BaseModel):
            answer: str = utils.Field(..., title="The answer to the question. The options can be only 'Yes' or 'No'")

        # Create a prompt template for relevance checking
        relevance_prompt = utils.PromptTemplate(
            input_variables=["query", "feedback_query", "doc_content", "feedback_response"],
            template="""
            Determine if the following feedback response is relevant to the current query and document content.
            You are also provided with the Feedback original query that was used to generate the feedback response.
            Current query: {query}
            Feedback query: {feedback_query}
            Document content: {doc_content}
            Feedback response: {feedback_response}

            Is this feedback relevant? Respond with only 'Yes' or 'No'.
            """
        )
        llm = utils.ChatOpenAI(temperature=0, model_name="gpt-4.1-nano", max_tokens=4000)

        # Create an LLMChain for relevance checking
        relevance_chain = relevance_prompt | llm.with_structured_output(self.Response)

        # Initialize relevance_score for all documents if not already present
        for doc in docs:
            if 'relevance_score' not in doc.metadata:
                doc.metadata['relevance_score'] = 1.0  # Default score

        for doc in docs:
            relevant_feedback = []

            for feedback in feedback_data:
                # Use LLM to check relevance
                input_data = {
                    "query": query,
                    "feedback_query": feedback['query'],
                    "doc_content": doc.page_content,
                    "feedback_response": feedback['response']
                }
                result = relevance_chain.invoke(input_data).answer

                if result == 'yes':
                    relevant_feedback.append(feedback)

            # Adjust the relevance score based on feedback
            if relevant_feedback:
                avg_relevance = sum(f['relevance'] for f in relevant_feedback) / len(relevant_feedback)
                doc.metadata['relevance_score'] *= (avg_relevance / 3)  # Assuming a 1-5 scale, 3 is neutral

        # Re-rank documents based on adjusted scores
        return sorted(docs, key=lambda x: x.metadata['relevance_score'], reverse=True)

    def fine_tune_index(
        self,
        feedback_data: utils.List[utils.Dict[str, utils.Any]],
        texts: utils.List[str],
        collection_name,
        path,
        database
    ) -> utils.Any:
        """
        fine tune the vector index to include also queries + answers that received good feedbacks
        """
        # Filter high-quality feedback
        good_responses = [f for f in feedback_data if f['relevance'] >= 4 and f['quality'] >= 4]

        # Extract queries and responses, and create new documents
        additional_texts = []
        for f in good_responses:
            combined_text = f['query'] + " " + f['response']
            additional_texts.append(combined_text)

        # Make the list a string
        additional_texts = " ".join(additional_texts)

        # Create a new index with original and high-quality texts
        all_texts = texts + additional_texts
        new_vectorstore = utils.get_vectorstore(
            collectionName=collection_name,
            path=path,
            cleaned_texts=all_texts,
            database=database
            )

        return new_vectorstore


########################################
#  reranking functions
########################################
class BaseReranker():
    """Base class for rerankers"""

    class Retriever(utils.BaseRetriever, utils.BaseModel):
        """Base retriever class"""
        vectorstore: utils.Any = utils.Field(description="Vector store for initial retrieval")
        parent_: utils.Any = utils.Field(description="Parent reranker object")

        class Config:
            arbitrary_types_allowed = True

        def _get_relevant_documents(self, query: str, num_docs=2) -> utils.List[utils.Document]:
            # Should be implemented by subclasses
            raise NotImplementedError("Subclasses must implement this method")

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.qa_chain = None
        self.retriever = self.Retriever(vectorstore=self.vectorstore, parent_=self)

    def get_retrieved_documents(self, query, num_docs=10):
        docs = self.retriever._get_relevant_documents(query, num_docs=num_docs)
        return docs

    def get_qa_chain(self, modelSetting={"temperature": 0, "model_name": "gpt-4.1-nano"}):
        llm = utils.ChatOpenAI(**modelSetting)
        self.qa_chain = utils.RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
        )
        return self.qa_chain

    def query_qa_chain(self, query):
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Call get_qa_chain() first.")

        results = self.qa_chain.invoke(query)
        answer = results["result"]
        source_documents = results["source_documents"]
        return answer, source_documents


class LLMReranker(BaseReranker):
    """
    LLM Method: Use prompts to ask the LLM to rate document relevance.
    """
    class LLMRetriever(utils.BaseRetriever, utils.BaseModel):
        """
        Create a custom retriever based on LLMReranker.
        """
        vectorstore: utils.Any = utils.Field(description="Vector store for initial retrieval")

        class Config:
            arbitrary_types_allowed = True

        def _get_relevant_documents(self, query: str, num_docs=2) -> utils.List[utils.Document]:
            initial_docs = self.vectorstore.similarity_search(query, k=30)
            return self.parent_.rerank_documents(query, initial_docs, top_n=num_docs)

    def __init__(self, vectorstore):
        super().__init__(vectorstore)
        self.retriever = self.LLMRetriever(vectorstore=self.vectorstore, parent_=self)

    def rerank_documents(self, query: str, docs: utils.List[utils.Document], top_n: int = 3) -> utils.List[utils.Document]:
        class RatingScore(utils.BaseModel):
            relevance_score: float = utils.Field(..., description="The relevance score of a document to a query.")

        prompt_template = utils.PromptTemplate(
            input_variables=["query", "doc"],
            template="""On a scale of 1-10, rate the relevance of the following document to the query. Consider the specific context and intent of the query, not just keyword matches.
            Query: {query}
            Document: {doc}
            Relevance Score:"""
        )

        llm = utils.ChatOpenAI(temperature=0, model_name="gpt-4.1-nano", max_tokens=4000)
        llm_chain = prompt_template | llm.with_structured_output(RatingScore)

        scored_docs = []
        for doc in docs:
            input_data = {"query": query, "doc": doc.page_content}
            score = llm_chain.invoke(input_data).relevance_score
            try:
                score = float(score)
            except ValueError:
                score = 0  # Default score if parsing fails
            scored_docs.append((doc, score))

        reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in reranked_docs[:top_n]]


class CrossEncoderReranker(BaseReranker):
    """
    Feed query-document pairs directly into the model.
    """
    class CrossEncoderRetriever(utils.BaseRetriever, utils.BaseModel):
        vectorstore: utils.Any = utils.Field(description="Vector store for initial retrieval")
        cross_encoder: utils.Any = utils.Field(description="Cross-encoder model for reranking")
        k: int = utils.Field(default=5, description="Number of documents to retrieve initially")
        rerank_top_k: int = utils.Field(default=3, description="Number of documents to return after reranking")

        class Config:
            arbitrary_types_allowed = True

        def _get_relevant_documents(self, query: str, num_docs=None) -> utils.List[utils.Document]:
            rerank_top_k = num_docs if num_docs is not None else self.rerank_top_k

            # Initial retrieval
            initial_docs = self.vectorstore.similarity_search(query, k=self.k)

            # Prepare pairs for cross-encoder
            pairs = [[query, doc.page_content] for doc in initial_docs]

            # Get cross-encoder scores
            scores = self.cross_encoder.predict(pairs)

            # Sort documents by score
            scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)

            # Return top reranked documents
            return [doc for doc, _ in scored_docs[:rerank_top_k]]

    def __init__(self, vectorstore):
        super().__init__(vectorstore)
        self.cross_encoder = utils.CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.retriever = self.CrossEncoderRetriever(
            vectorstore=self.vectorstore,
            cross_encoder=self.cross_encoder,
            k=10,
            rerank_top_k=5)
