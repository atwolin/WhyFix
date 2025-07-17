# For functionas that use langchain
import os, re
import random
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
import yaml
import json
import hashlib

# Preprocessing
import fitz
from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

# Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Vector Store and Retrieval
# from langchain.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from langchain.chains import RetrievalQA
from langchain.chains.summarize.chain import load_summarize_chain
from langchain_core.retrievers import BaseRetriever
from sentence_transformers import CrossEncoder

# Model
import openai
from langchain_openai import ChatOpenAI
import cohere
from langchain_core.runnables import chain

# Prompt
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List, Any, Dict, Tuple

from openai import RateLimitError


import asyncio
# import nest_asyncio


import textwrap
from enum import Enum

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["CO_API_KEY"] = os.getenv("CO_API_KEY")

folderPath_persist = "/home/nlplab/atwolin/thesis/data/L2-knowledge/vectorstore"


class SupportMaterial():
    """
    Class to handle the support material for the retriever.
    """

    def __init__(self, topic, embeddingType):
        # L2 knowledge
        self.folderPath_l2_main = "/home/nlplab/atwolin/thesis/data/L2-knowledge"

        self.folderPath_l2_chroma = os.path.join(self.folderPath_l2_main, "vectorstore")
        self.folderPath_l2_faiss = os.path.join(self.folderPath_l2_main, "faiss_index")

        self.folderPath_all = os.path.join(self.folderPath_l2_main, "all")
        self.folderPath_collocation = os.path.join(self.folderPath_l2_main, "collocation")
        self.folderPath_lexicon = os.path.join(self.folderPath_l2_main, "causes-lexical-errors")
        self.folderPath_writing = os.path.join(self.folderPath_l2_main, "academic-writing")

        self.folderPath_result = os.path.join(self.folderPath_l2_main, "results")
        self.filePath_result_drag = os.path.join(self.folderPath_result, f"result_drag_{embeddingType}_{topic}.json")
        self.filePath_result_iter_drag = os.path.join(self.folderPath_result, f"result_iter_drag_{embeddingType}_{topic}.json")
        # ---- ground truths ----
        self.filePath_l2_ground_truth_academic_writing = os.path.join(self.folderPath_l2_main, "ground_truth_academic_writing.txt")
        self.filePath_l2_ground_truth_causes = os.path.join(self.folderPath_l2_main, "ground_truth_causes.txt")


        # self.filePath_eval_drag = os.path.join(self.folderPath_result, f"eval_drag_{embeddingType}_{topic}.json")
        # self.filePath_eval_iter_drag = os.path.join(self.folderPath_result, f"eval_iter_drag_{embeddingType}_{topic}.json")
        self.folderPath_eval = os.path.join(self.folderPath_l2_main, "evaluations")
        os.makedirs(self.folderPath_eval, exist_ok=True)
        self.filePath_eval_drag = os.path.join(self.folderPath_eval, f"eval_drag_{embeddingType}_{topic}.json")
        self.filePath_eval_iter_drag = os.path.join(self.folderPath_eval, f"eval_iter_drag_{embeddingType}_{topic}.json")

        # Collocation collections
        self.folderPath_collocation_main = "/home/nlplab/atwolin/thesis/data/collocation-collections"

        self.folderPath_collocation_faiss = os.path.join(self.folderPath_collocation_main, "faiss_index")

        self.filePath_collocation_txt_raw = os.path.join(self.folderPath_collocation_main, "collocations_0518.txt")
        self.filePath_collocation_csv_raw = os.path.join(self.folderPath_collocation_main, "data_0516.csv")
        # self.filePath_collocation_csv_collection = os.path.join(self.folderPath_collocation_main, "collocation_collection.csv")
        self.filePath_collocation_full_csv = os.path.join(self.folderPath_collocation_main, "collocation_full.csv")
        self.filePath_collocation_full_txt = os.path.join(self.folderPath_collocation_main, "collocation_full.txt")
        self.filePath_collocation_simplified_csv = os.path.join(self.folderPath_collocation_main, "collocation_simplified.csv")
        self.filePath_collocation_simplified_txt = os.path.join(self.folderPath_collocation_main, "collocation_simplified.txt")

    def loadSupportMaterialList(self, path):
        """
        Load a list of support materials from the given path.
        """
        if os.path.exists(path):
            filePaths = os.listdir(path)
            supportMaterialList = []
            for filePath in filePaths:
                if filePath.endswith(".pdf"):
                    supportMaterialList.append(os.path.join(path, filePath))
            return supportMaterialList
        else:
            raise FileNotFoundError(f"Support material file not found at {path}")


class DictionaryInfo:
    word_learner: str
    lemmaWord_learner: str
    pos_learner: str
    level_learner: str
    definition_learner: str
    examples_learner: str
    in_akl_learner: bool
    word_editor: str
    lemmaWord_editor: str
    pos_editor: str
    level_editor: str
    definition_editor: str
    examples_editor: str
    in_akl_editor: bool


class L2KnowledgeInfo:
    causes_fifty: List[str]
    academic_writing_fifty: List[str]
    causes_eighty: List[str]
    academic_writing_eighty: List[str]


class CollocationInfo:
    collocations: str
    other_categories_json: str
    other_categories_formatted_json: str
########################################
#      printing functions
########################################
def pretify_print(data):
    """
    Pretty print the JSON data.
    """
    print(json.dumps(data, indent=4))


def word_wrap(string, n_chars=72):
    # Wrap a string at the next space after n_chars
    if len(string) < n_chars:
        return string
    else:
        return string[:n_chars].rsplit(' ', 1)[0] + '\n' + word_wrap(string[len(string[:n_chars].rsplit(' ', 1)[0])+1:], n_chars)


def show_context(context):
    """
    Display the contents of the provided context list.

    Args:
        context (list): A list of context items to be displayed.

    Prints each context item in the list with a heading indicating its position.
    """
    for i, c in enumerate(context):
        print(f"Context {i + 1}:")
        print(c)
        print("\n")


def print_chunk_segment(chunks, startIdx: int, endIdx: int):
    """
    Print the text content of a segment of the document

    Args:
        chunks (list): List of text or document chunks
        start_index (int): Start index of the segment
        end_index (int): End index of the segment (not inclusive)

    Returns:
        None

    Prints:
        The text content of the specified segment of the document
    """
    if type(chunks) == Document:
        chunks = [chunk.page_content for chunk in chunks]

    for i in range(startIdx, endIdx):
        print(f"\nChunk {i}")
        print(chunks[i])


########################################
#      read files functions
########################################
def load_yaml(path="/home/nlplab/atwolin/thesis/code/process/utils/retrieve_query.yaml"):
    experiment = {}
    with open(path, "r") as f:
        experiment = yaml.load(f, Loader=yaml.FullLoader)
    return experiment


def dump_yaml(path, data, writeMode="w"):
    with open(path, writeMode) as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, indent=2)


def read_pdf(filename):
    reader = PdfReader(filename)

    pdf_texts = [p.extract_text().strip() for p in reader.pages]

    # Filter the empty strings
    pdf_texts = [text for text in pdf_texts if text]
    return pdf_texts


def read_pdf_to_string(path):
    """
    Read a PDF document from the specified path and return its content as a string.

    Args:
        path (str): The file path to the PDF document.

    Returns:
        str: The concatenated text content of all pages in the PDF document.

    The function uses the 'fitz' library (PyMuPDF) to open the PDF document, iterate over each page,
    extract the text content from each page, and append it to a single string.
    """
    # Open the PDF document located at the specified path
    doc = fitz.open(path)
    content = ""
    # Iterate over each page in the document
    for page_num in tqdm(range(len(doc)), desc="Reading PDF pages"):
        # Get the current page
        page = doc[page_num]
        # Extract the text content from the current page and append it to the content string
        content += page.get_text()
    return content


def replace_t_with_space(list_of_documents):
    """
    Replaces all tab characters ('\t') with spaces in the page content of each document

    Args:
        list_of_documents: A list of document objects, each with a 'page_content' attribute.

    Returns:
        The modified list of documents with tab characters replaced by spaces.
    """
    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace("\t", " ")
    return list_of_documents


def sanitize_string(s: str) -> str:
    """
    Sanitize a string by replacing spaces (' '), dashes ('-'), and colons (':') with underscores.
    """
    # [-\s:]+ matches one or more occurrences of '-', ' ', or ':'
    return re.sub(r'[-\s:]+', '_', s)


########################################
#      chunking functions
########################################
def chunk_to_documents(path_doc, chunkSize=1000, chunkOverlap=200):
    """
    Split a given text into chunks of specified size using RecursiveCharacterTextSplitter.
    Args:
        path_doc (str): The path to the PDF document.
        chunkSize (int): The desired size of each text chunk.
        chunkOverlap (int): The amount of overlap between consecutive chunks.
    Returns:
        list: A list of Document objects representing the chunks of the PDF document.
    """
    # Load PDF documents
    loader = PyPDFLoader(path_doc)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunkSize, chunk_overlap=chunkOverlap, length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    cleaned_chunks = replace_t_with_space(chunks)

    return cleaned_chunks


def chunk_texts_into_sentences(texts):
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=0
    )
    # character_split_texts = character_splitter.split_text('\n\n'.join(texts))
    character_split_texts = character_splitter.split_text(texts)

    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)

    token_split_texts = []
    for text in character_split_texts:
        token_split_texts += token_splitter.split_text(text)

    return character_split_texts, token_split_texts


def chunk_to_pure_text(text, chunkSize=1000, chunkOverlap=200):
    """
    Split a given text in path_doc into chunks of specified size using RecursiveCharacterTextSplitter.

    Args:
        text (str): The input text to be split into chunks.
        chunkSize (int, optional): The maximum size of each chunk. Defaults to 1000.
        chunkOverlap (int, optional): The amount of overlap between consecutive chunks. Defaults to 200.

    Returns:
        list[str]: A list of text chunks.

    Example:
        >>> text = "This is a sample text to be split into chunks."
        >>> chunks = split_into_chunks(text, chunk_size=10)
        >>> print(chunks)
        ['This is a', 'sample', 'text to', 'be split', 'into', 'chunks.']
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunkSize, chunk_overlap=chunkOverlap, length_function=len)
    texts = text_splitter.create_documents([text])
    chunks = [text.page_content for text in texts]

    return chunks

########################################
#     embedding functions
########################################
# Enum class representing different embedding providers
class EmbeddingProvider(Enum):
    OPENAI = "openai"
    HG = "huggingface"


def get_langchain_embedding_provider(provider: EmbeddingProvider, embeddingSettings: dict = None):
    """
    Returns an embedding provider based on the specified provider and model ID.

    Args:
        provider (EmbeddingProvider): The embedding provider to use.
        model_id (str): Optional -  The specific embeddings model ID to use .

    Returns:
        A LangChain embedding provider instance.

    Raises:
        ValueError: If the specified provider is not supported.
    """
    if provider.value == EmbeddingProvider.OPENAI.value:
        print(f"Embedding settings: {embeddingSettings}")
        return OpenAIEmbeddings(**embeddingSettings) if embeddingSettings else OpenAIEmbeddings(model="text-embedding-3-large")
    elif provider.value == EmbeddingProvider.HG.value:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(**embeddingSettings) if embeddingSettings else HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


########################################
#     vectorstore functions
########################################
# Enum class representing different vector databases
class VectorDatabase(Enum):
    CHROMA = "chroma"
    FAISS = "faiss"


def get_vectorstore(
    collectionName, path,
    cleaned_docs: list = [Document(page_content=".", metadata={"source": "https://example.com"})],
    database=VectorDatabase.FAISS,
    embeddingProvider=EmbeddingProvider.OPENAI,
    embeddingSettings: dict = {}
):
    # Print
    print(f"{'=' * 50}\nDatabase: {database}, Embedding provider: {embeddingProvider}, Collection name: {collectionName}\n{'=' * 50}")
    # Print end

    vectorstore = None
    embeddings = get_langchain_embedding_provider(embeddingProvider, embeddingSettings=embeddingSettings)

    print(f"Embedding model: {embeddings.model}")
    print('=' * 50)

    if database.value == VectorDatabase.CHROMA.value:
        vectorstore = Chroma(
            collection_name=collectionName,
            embedding_function=embeddings,
            persist_directory=path
            )

    elif database.value == VectorDatabase.FAISS.value:
        if os.path.exists(path):
            vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
            print(f"Loaded a FAISS vectorstore in {path}")
        else:
            os.makedirs(path, exist_ok=True)
            vectorstore = FAISS.from_documents(cleaned_docs, embedding=embeddings)
            vectorstore.save_local(path)
            print(f"Created a new FAISS vectorstore in {path}")

    return vectorstore


def load_doc_to_vectorstore(path, vectorstore, docs: Document):
    """
    Loads documents to a vector store with deduplication based on document sources.

    Args:
        vectorstore: The vector store instance
        docs: List of documents to add
        database: VectorDatabase enum type

    Returns:
        Updated vector store instance
    """
    # Create a dictionary to track page_content that have already been added
    existing_sources = set()

    # Get existing documents if available and gather their page_content
    for page_content in vectorstore.docstore._dict:
        doc = vectorstore.docstore._dict[page_content]
        if hasattr(doc, "page_content"):
            existing_sources.add(doc.page_content)

    # Filter out documents with duplicate page_content
    new_docs = []
    for doc in docs:
        if doc.page_content not in existing_sources:
            new_docs.append(doc)
            existing_sources.add(doc.page_content)

    # Only add new documents
    if new_docs:
        vectorstore.add_documents(documents=new_docs)
        vectorstore.save_local(path)
        # print(f"{len(new_docs)} new documents added to vectorstore")
    else:
        print("No new documents to add (all were duplicates)")

    return vectorstore


def encode_pdf(path_doc, path_vectorstore, vectorstore, database=VectorDatabase.FAISS, chunking_settings: dict = {"chunk_size": 1000, "chunk_overlap": 200, "length_function": len}):
    """
    Encodes a PDF book into a vector store using OpenAI embeddings.

    Args:
        path: The path to the PDF file.
        chunkSize: The desired size of each text chunk.
        chunkOverlap: The amount of overlap between consecutive chunks.

    Returns:
        A Chroma vector store containing the encoded book content.
    """
    # Load PDF documents
    loader = PyPDFLoader(path_doc)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(**chunking_settings)
    docs = text_splitter.split_documents(documents)
    cleaned_docs = replace_t_with_space(docs)

    # Create embedding function
    source_name = path_doc.split("/")[-1].split("_")[0].strip()
    source_name = sanitize_string(source_name)

    # Create IDs for chunks
    for i, doc in enumerate(cleaned_docs):
        doc.metadata["id"] = f"{source_name}_{i}"
        # if i == 0:
        #     print(f"First chunk ID: {doc.metadata['id']}")

    # print(f"{'=' * 50}\nFinished {source_name} chunking")

    vectorstore = load_doc_to_vectorstore(path_vectorstore, vectorstore, cleaned_docs)
    return vectorstore


########################################
#     retrieval functions
########################################
@chain
def retriever_with_score(inputs) -> List[Document]:
    vectorstore = inputs["vectorstore"]
    query = inputs["query"]
    k = inputs.get("k", 5)  # Default to 5 if not provided
    docs, scores = zip(*vectorstore.similarity_search_with_score(query, k=k))
    for doc, score in zip(docs, scores):
        doc.metadata["score"] = score
    return list(docs)


def retrieve_context_per_question(question, retriever):
    """
    Retrieves relevant context and unique URLs for a given question using the chunks query retriever.

    Args:
        question: The question for which to retrieve context and URLs.
        retriever: The retriever object to use for querying.

    Returns:
        A tuple containing:
        - A string with the concatenated content of relevant documents.
        - A list of Document objects containing the relevant documents.
    """
    # Retrieve relevant documents for the given question
    docs = retriever.invoke(question)

    # Concatenate document contents
    context = [doc.page_content for doc in docs]
    return context, docs


########################################
#     reranking functions
########################################
def rerank_pure_text_chunks(query: str, chunks: List[str]):
    """
    Use Cohere Rerank API to rerank the search results

    Args:
        query (str): The search query
        chunks (list): List of chunks to be reranked

    Returns:
        similarity_scores (list): List of similarity scores for each chunk
        chunk_values (list): List of relevance values (fusion of rank and similarity) for each chunk
    """
    model = "rerank-english-v3.0"
    client = cohere.Client()
    decay_reate = 30

    rerank_results = client.rerank(model=model, query=query, documents=chunks)
    results = rerank_results.results
    reranked_indices = [result.index for result in results]
    reranked_similarity_scores = [result.relevance_score for result in results]  # in order of reranked_indices

    # Convert back to order of original documents and calculate the chunk values
    similarity_scores = [0] * len(chunks)
    chunk_values = [0] * len(chunks)
    for i, index in enumerate(reranked_indices):
        absolute_relevance_value = transform(reranked_similarity_scores[i])
        similarity_scores[index] = absolute_relevance_value
        chunk_values[index] = np.exp(-i/decay_reate) * absolute_relevance_value  # decay the relevance value based on the rank

    return results, similarity_scores, chunk_values


########################################
#     llm functions
########################################
