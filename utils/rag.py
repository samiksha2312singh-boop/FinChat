
"""
RAG System for SEC Filing Analysis
Handles document ingestion, chunking, embedding, and retrieval
"""

# Fix for ChromaDB SQLite issue - MUST BE FIRST
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import chromadb
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List


@st.cache_resource
def init_chromadb():
    """Initialize ChromaDB client with persistent storage"""
    client = chromadb.PersistentClient(path="./data/sec_filings_db")
    return client


def create_collection_for_ticker(ticker: str):
    """Create or retrieve collection for specific ticker"""
    client = init_chromadb()
    collection_name = f"{ticker.lower()}_filings"
    
    try:
        collection = client.get_collection(name=collection_name)
    except:
        collection = client.create_collection(name=collection_name)
    
    return collection


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks
    
    Args:
        text: Input text to chunk
        chunk_size: Target size of each chunk
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period > chunk_size * 0.7:
                end = start + last_period + 1
                chunk = text[start:end]
        
        if chunk.strip():
            chunks.append(chunk.strip())
        
        start = end - overlap
    
    return chunks


def ingest_filing_to_rag(filing_data: Dict, ticker: str):
    """
    Process SEC filing and store in vector database
    
    Args:
        filing_data: Dictionary with filing sections
        ticker: Stock ticker symbol
    
    Returns:
        ChromaDB collection
    """
    collection = create_collection_for_ticker(ticker)
    
    # Check if already ingested
    if collection.count() > 0:
        st.info(f"Filing already loaded for {ticker}")
        return collection
    
    # Prepare chunks
    all_chunks = []
    all_metadatas = []
    
    for section_name, content in filing_data['sections'].items():
        chunks = chunk_text(content, chunk_size=500)
        
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({
                'section': section_name,
                'ticker': ticker,
                'filename': filing_data.get('filename', 'unknown')
            })
    
    if not all_chunks:
        st.warning("No content to ingest")
        return collection
    
    # Generate embeddings
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=st.secrets["OPENAI_API_KEY"]
    )
    
    try:
        embeddings = embeddings_model.embed_documents(all_chunks)
        ids = [f"{ticker}_chunk_{i}" for i in range(len(all_chunks))]
        
        # Add to ChromaDB
        collection.add(
            documents=all_chunks,
            embeddings=embeddings,
            metadatas=all_metadatas,
            ids=ids
        )
        
        st.success(f"✅ Ingested {len(all_chunks)} chunks for {ticker}")
        
    except Exception as e:
        st.error(f"Error ingesting filing: {str(e)}")
    
    return collection


def search_filing(ticker: str, query: str, n_results: int = 5) -> List[Dict]:
    """
    Search SEC filing using semantic search
    
    Args:
        ticker: Stock ticker
        query: Search query
        n_results: Number of results to return
    
    Returns:
        List of relevant chunks with metadata
    """
    try:
        collection = create_collection_for_ticker(ticker)
        
        if collection.count() == 0:
            return []
        
        # Generate query embedding
        embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=st.secrets["OPENAI_API_KEY"]
        )
        
        query_embedding = embeddings_model.embed_query(query)
        
        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        chunks = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                chunks.append({
                    'text': results['documents'][0][i],
                    'section': results['metadatas'][0][i].get('section', 'Unknown'),
                    'distance': results['distances'][0][i],
                    'relevance': 1 - results['distances'][0][i]
                })
        
        return chunks
        
    except Exception as e:
        st.error(f"RAG search error: {str(e)}")
        return []


def clear_filing_data(ticker: str):
    """Clear all stored data for a ticker"""
    try:
        client = init_chromadb()
        collection_name = f"{ticker.lower()}_filings"
        client.delete_collection(name=collection_name)
        st.success(f"Cleared data for {ticker}")
    except Exception as e:
        st.error(f"Error clearing data: {str(e)}")
