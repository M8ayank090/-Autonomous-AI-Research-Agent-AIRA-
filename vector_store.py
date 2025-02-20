from typing import List, Dict, Any
import numpy as np
from pathlib import Path
import faiss
import chromadb
from weaviate import Client
from config import config
import torch
from transformers import AutoTokenizer, AutoModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, config):
        self.config = config
        self.store_type = config.VECTOR_STORE_TYPE
        self.store = self._initialize_store()
        self.embedding_model = self._load_embedding_model()
        
    def _load_embedding_model(self):
        """Load the embedding model based on config"""
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
        model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")
        return {'tokenizer': tokenizer, 'model': model}
        
    def _initialize_store(self):
        """Initialize the vector store based on config"""
        if self.store_type == "FAISS":
            return self._init_faiss()
        elif self.store_type == "ChromaDB":
            return self._init_chromadb()
        elif self.store_type == "Weaviate":
            return self._init_weaviate()
        else:
            raise ValueError(f"Unsupported vector store type: {self.store_type}")
            
    def _init_faiss(self):
        """Initialize FAISS index"""
        dimension = 1024  # BGE-large embedding dimension
        index = faiss.IndexFlatL2(dimension)
        return index
        
    def _init_chromadb(self):
        """Initialize ChromaDB"""
        client = chromadb.Client()
        collection = client.create_collection(
            name="research_papers",
            metadata={"description": "Research paper embeddings"}
        )
        return collection
        
    def _init_weaviate(self):
        """Initialize Weaviate"""
        client = Client("http://localhost:8080")
        return client
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using the loaded model"""
        tokenizer = self.embedding_model['tokenizer']
        model = self.embedding_model['model']
        
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use CLS token embedding
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        return embeddings
        
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to vector store"""
        try:
            for doc in documents:
                # Create combined text for embedding
                text = f"{doc['title']} {doc['summary']}"
                embedding = self.get_embedding(text)
                
                if self.store_type == "FAISS":
                    self.store.add(embedding)
                    
                elif self.store_type == "ChromaDB":
                    self.store.add(
                        embeddings=embedding.tolist(),
                        documents=[text],
                        metadatas=[{
                            'title': doc['title'],
                            'authors': doc['authors'],
                            'published': doc['published']
                        }],
                        ids=[doc['arxiv_id']]
                    )
                    
                elif self.store_type == "Weaviate":
                    self.store.data_object.create(
                        class_name="Research",
                        data_object={
                            'title': doc['title'],
                            'text': text,
                            'authors': doc['authors'],
                            'published': doc['published']
                        },
                        vector=embedding.tolist()
                    )
                    
            logger.info(f"Added {len(documents)} documents to {self.store_type}")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        query_embedding = self.get_embedding(query)
        
        if self.store_type == "FAISS":
            D, I = self.store.search(query_embedding, k)
            return [{'distance': float(d), 'index': int(i)} for d, i in zip(D[0], I[0])]
            
        elif self.store_type == "ChromaDB":
            results = self.store.query(
                query_embeddings=query_embedding.tolist(),
                n_results=k
            )
            return results
            
        elif self.store_type == "Weaviate":
            results = (
                self.store.query
                .get("Research")
                .near_vector({
                    'vector': query_embedding.tolist()
                })
                .with_limit(k)
                .do()
            )
            return results
