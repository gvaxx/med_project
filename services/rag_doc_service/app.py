from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import os
from datetime import datetime
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Document Service")

# Configure data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

class Document(BaseModel):
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentResponse(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    similarity: Optional[float] = None

class SearchQuery(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1, le=10)
    filter_metadata: Optional[Dict[str, Any]] = None

class VectorStore:
    def __init__(self, collection_name: str = "medical_documents"):
        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(path=os.path.join(DATA_DIR, "chroma"))
        
        # Use sentence-transformers embedding function
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name='all-MiniLM-L6-v2'
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_fn
            )
            logger.info(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_fn,
                metadata={"description": "Medical documents collection"}
            )
            logger.info(f"Created new collection: {collection_name}")

    def add_document(self, content: str, metadata: Dict[str, Any]) -> str:
        """Add a document to the vector store"""
        try:
            # Generate unique ID
            doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            # check if the metadata has lists in it
            for key, value in metadata.items():
                if isinstance(value, list):
                    metadata[key] = ', '.join(value)
            # Add timestamp to metadata
            metadata = {
                **metadata,
                "added_at": datetime.now().isoformat(),
                "doc_id": doc_id
            }
            
            # Add document to ChromaDB
            self.collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            logger.info(f"Added document with ID: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            raise

    def search(
        self, 
        query: str, 
        top_k: int = 3, 
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentResponse]:
        """Search for similar documents"""
        try:
            # Prepare filter if provided
            where = filter_metadata if filter_metadata else None
            
            # Search in ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where
            )
            
            # Format results
            documents = []
            for idx in range(len(results['ids'][0])):
                doc = DocumentResponse(
                    id=results['ids'][0][idx],
                    content=results['documents'][0][idx],
                    metadata=results['metadatas'][0][idx],
                    similarity=float(results['distances'][0][idx])
                )
                documents.append(doc)
            
            return documents

        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            raise

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the vector store"""
        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"Deleted document with ID: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False

    def get_all_documents(self) -> List[DocumentResponse]:
        """Get all documents from the store"""
        try:
            results = self.collection.get()
            documents = []
            
            if not results['ids']:
                return documents
                
            for idx in range(len(results['ids'])):
                doc = DocumentResponse(
                    id=results['ids'][idx],
                    content=results['documents'][idx],
                    metadata=results['metadatas'][idx],
                    similarity=None
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error getting documents: {str(e)}")
            raise

# Initialize vector store
vector_store = VectorStore()

@app.get("/")
async def root():
    return {"message": "RAG Document Service"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/documents", response_model=DocumentResponse)
async def add_document(document: Document):
    """Add a new document to the vector store"""
    try:
        doc_id = vector_store.add_document(document.content, document.metadata)
        return DocumentResponse(
            id=doc_id,
            content=document.content,
            metadata=document.metadata
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=List[DocumentResponse])
async def list_documents():
    """List all documents"""
    try:
        return vector_store.get_all_documents()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document from the vector store"""
    if vector_store.delete_document(doc_id):
        return {"message": f"Document {doc_id} deleted"}
    raise HTTPException(status_code=404, detail="Document not found")

@app.post("/search", response_model=List[DocumentResponse])
async def search_documents(query: SearchQuery):
    """Search for similar documents"""
    try:
        results = vector_store.search(
            query.query, 
            query.top_k,
            query.filter_metadata
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))