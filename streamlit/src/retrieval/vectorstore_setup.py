import os
import re
from enum import Enum
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import chain


########################################
#      chunking functions
########################################
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
#     embedding functions
########################################
class EmbeddingProvider(Enum):
    """Enum class representing different embedding providers"""
    OPENAI = "openai"
    HG = "huggingface"


def get_langchain_embedding_provider(provider: EmbeddingProvider, kwargs: dict = None):
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
        return OpenAIEmbeddings(model=kwargs["model_name"]) if kwargs else OpenAIEmbeddings(model="text-embedding-3-small")
    # elif provider.value == EmbeddingProvider.HG.value:
    # from langchain_huggingface import HuggingFaceEmbeddings
    #     return HuggingFaceEmbeddings(model=kwargs["model_name"]) if kwargs else HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


########################################
#     vectorstore functions
########################################
class VectorDatabase(Enum):
    """Enum class representing different vector databases"""
    CHROMA = "chroma"
    FAISS = "faiss"


def get_vectorstore(
    collectionName, path,
    cleaned_docs: list = [Document(page_content=".", metadata={"source": "https://example.com"})],
    database=VectorDatabase.FAISS,
    embeddingProvider=EmbeddingProvider.OPENAI,
    kwargs: dict = {}
):
    # Print
    print(f"{'=' * 50}\nDatabase: {database}, Embedding provider: {embeddingProvider}, Collection name: {collectionName}\n{'=' * 50}")
    # Print end

    vectorstore = None
    embeddings = get_langchain_embedding_provider(embeddingProvider, kwargs=kwargs)

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
    else:
        print("No new documents to add (all were duplicates)")

    return vectorstore


def encode_pdf_to_vectorstore(path_doc, path_vectorstore, vectorstore, database=VectorDatabase.FAISS, chunking_settings: dict = {"chunk_size": 1000, "chunk_overlap": 200, "length_function": len}):
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
