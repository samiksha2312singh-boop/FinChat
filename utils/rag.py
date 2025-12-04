"""
RAG System for SEC Filing Analysis
Handles document ingestion, chunking, embedding, retrieval, and reranking
"""

# Fix for ChromaDB SQLite issue - MUST BE FIRST
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import chromadb
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List


# ============================================================================
# INITIALIZERS (CACHED)
# ============================================================================

@st.cache_resource
def init_chromadb():
    """Initialize ChromaDB client with persistent storage."""
    client = chromadb.PersistentClient(path="./data/sec_filings_db")
    return client


@st.cache_resource
def get_embeddings_model():
    """Return a cached OpenAI embeddings model instance."""
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=st.secrets["OPENAI_API_KEY"]
    )


# ============================================================================
# COLLECTION MANAGEMENT
# ============================================================================

def create_collection_for_ticker(ticker: str):
    """Create or retrieve collection for specific ticker."""
    client = init_chromadb()
    collection_name = f"{ticker.lower()}_filings"
    
    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        collection = client.create_collection(name=collection_name)
    
    return collection


# ============================================================================
# CHUNKING
# ============================================================================

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text to chunk
        chunk_size: Target size of each chunk (in characters)
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


# ============================================================================
# INGESTION
# ============================================================================

def ingest_filing_to_rag(filing_data: Dict, ticker: str):
    """
    Process SEC filing and store in vector database.
    
    Args:
        filing_data: Dictionary with filing sections
        ticker: Stock ticker symbol
    
    Returns:
        ChromaDB collection
    """
    collection = create_collection_for_ticker(ticker)
    
    # Check if already ingested (allow re-ingestion for updates)
    if collection.count() > 0:
        st.info(f"Filing data exists for {ticker}. Adding new data...")
    
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
    
    # Generate embeddings (cached model)
    embeddings_model = get_embeddings_model()
    
    try:
        embeddings = embeddings_model.embed_documents(all_chunks)
        
        # Generate unique IDs
        import time
        timestamp = int(time.time())
        ids = [f"{ticker}_chunk_{timestamp}_{i}" for i in range(len(all_chunks))]
        
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


# ============================================================================
# BASIC SEMANTIC SEARCH
# ============================================================================

def search_filing(ticker: str, query: str, n_results: int = 5) -> List[Dict]:
    """
    Search SEC filing using semantic search (basic retrieval).
    
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
            # No filing data for this ticker yet
            return []
        
        # Generate query embedding (cached model)
        embeddings_model = get_embeddings_model()
        query_embedding = embeddings_model.embed_query(query)
        
        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        chunks = []
        if results.get('documents') and results['documents'][0]:
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


# ============================================================================
# RERANKING PIPELINE
# ============================================================================

def rerank_chunks(ticker: str, query: str, n_results: int = 3) -> List[Dict]:
    """
    Search and rerank chunks using LLM for better relevance.
    
    This implements the full RAG + Reranking pipeline:
    1. Retrieve top N chunks using vector search
    2. Use LLM to score each chunk's relevance
    3. Return top K highest-scored chunks
    
    Args:
        ticker: Stock ticker
        query: Search query
        n_results: Final number of results after reranking
    
    Returns:
        List of reranked chunks with scores
    """
    from langchain_fireworks import ChatFireworks
    import json
    
    try:
        # Step 1: RETRIEVE - Get more chunks than needed
        initial_n = min(10, n_results * 3)  # Get ~3x more for reranking
        initial_chunks = search_filing(ticker, query, n_results=initial_n)
        
        if not initial_chunks or len(initial_chunks) == 0:
            return []
        
        # If we got fewer chunks than requested, just return them
        if len(initial_chunks) <= n_results:
            return initial_chunks
        
        # Step 2: RERANK - Use LLM to score relevance
        llm = ChatFireworks(
            model="accounts/fireworks/models/llama-v3p3-70b-instruct",
            api_key=st.secrets["FIREWORKS_API_KEY"],
            temperature=0,  # Deterministic for scoring
            max_tokens=100
        )
        
        # Prepare chunks for scoring
        chunks_text = "\n\n".join([
            f"Chunk {i+1} [Section: {chunk['section']}]:\n{chunk['text'][:400]}..."
            for i, chunk in enumerate(initial_chunks)
        ])
        
        rerank_prompt = f"""Score each chunk's relevance to the query on a scale of 1-10.

Consider:
- How directly it answers the query
- Specificity of information
- Completeness of data

Query: "{query}"

Chunks:
{chunks_text}

Return ONLY a JSON array of integer scores: [score1, score2, score3, ...]

Example: [8, 3, 9, 2, 7, 6]
"""
        
        response = llm.invoke(rerank_prompt)
        scores_text = response.content.strip()
        
        # Clean up JSON response
        scores_text = scores_text.replace('```json', '').replace('```', '').strip()
        
        try:
            scores = json.loads(scores_text)
        except Exception:
            # Fallback if LLM didn't return valid JSON
            st.warning("Reranking failed, using vector search results")
            return initial_chunks[:n_results]
        
        # Step 3: Add scores and sort
        for i, chunk in enumerate(initial_chunks[:len(scores)]):
            chunk['rerank_score'] = scores[i]
        
        # Sort by rerank score (highest first)
        reranked = sorted(
            initial_chunks[:len(scores)], 
            key=lambda x: x.get('rerank_score', 0), 
            reverse=True
        )
        
        # Return top k
        return reranked[:n_results]
        
    except Exception as e:
        st.error(f"Reranking error: {str(e)}")
        # Fallback to basic search
        return search_filing(ticker, query, n_results)


# ============================================================================
# MAINTENANCE
# ============================================================================

def clear_filing_data(ticker: str):
    """Clear all stored filing data for a ticker."""
    try:
        client = init_chromadb()
        collection_name = f"{ticker.lower()}_filings"
        client.delete_collection(name=collection_name)
        st.success(f"Cleared data for {ticker}")
    except Exception as e:
        st.error(f"Error clearing data: {str(e)}")
