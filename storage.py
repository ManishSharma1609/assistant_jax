import os
from dotenv import load_dotenv
import json
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.storage import InMemoryStore
from langchain.retrievers import MultiVectorRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv()
class JAXFAISSRetriever:
    def __init__(self, knowledge_base_path: str):
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  
        
        # Load knowledge base
        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
        
        # Initialize vector stores
        self.vectorstore, self.docstore = self._create_vector_stores()
        
        # Create retriever
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            id_key="doc_id"
        )
    
    def _load_knowledge_base(self, path: str) -> List[Dict]:
        """Load preprocessed knowledge base from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _create_vector_stores(self):
        """Create FAISS vector store and document store"""
        # Prepare documents
        documents = []
        metadatas = []
        ids = []
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        for doc in self.knowledge_base:
            # Split document content
            splits = text_splitter.split_text(doc['content'])
            for i, split in enumerate(splits):
                doc_id = f"{doc['id']}_{i}"
                documents.append(split)
                metadatas.append({
                    'type': 'documentation',
                    'title': doc['title'],
                    'source': doc['path'],
                    'doc_id': doc['id'],
                    'chunk_id': i
                })
                ids.append(doc_id)
            
            # Add code blocks
            for i, code_block in enumerate(doc['code_blocks']):
                code_id = f"code_{doc['id']}_{i}"
                documents.append(code_block)
                metadatas.append({
                    'type': 'code',
                    'title': doc['title'],
                    'source': doc['path'],
                    'doc_id': doc['id'],
                    'code_block_id': i
                })
                ids.append(code_id)
        
        # Create FAISS vector store
        vectorstore = FAISS.from_texts(
            texts=documents,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        
        # Create document store - CORRECTED IMPLEMENTATION
        docstore = InMemoryStore()
        # Convert to list of tuples as required by mset
        docstore.mset([(doc['id'], doc) for doc in self.knowledge_base])
        
        return vectorstore, docstore
    
    def query(self, question: str, include_code: bool = True, top_k: int = 3):
        """Execute a query with optional code filtering"""
        if include_code:
            docs = self.vectorstore.similarity_search(
                question,
                k=top_k,
                filter=lambda meta: meta.get('type') == 'code'
            )
        else:
            docs = self.vectorstore.similarity_search(
                question,
                k=top_k,
                filter=lambda meta: meta.get('type') == 'documentation'
            )
        
        # Convert to LangChain Document objects
        lc_docs = [
            Document(
                page_content=doc.page_content,
                metadata=doc.metadata
            ) for doc in docs
        ]
        
        # Get full documents for context
        doc_ids = list(set([doc.metadata['doc_id'] for doc in docs]))
        full_docs = [doc for doc in self.docstore.mget(doc_ids) if doc is not None]
        
        return {
            "relevant_chunks": lc_docs,
            "source_documents": full_docs
        }
    
    def save_index(self, path: str):
        """Save FAISS index to disk"""
        self.vectorstore.save_local(path)
    
    @classmethod
    def load_index(cls, path: str, knowledge_base_path: str):
        """Load FAISS index from disk"""
        retriever = cls(knowledge_base_path)
        retriever.vectorstore = FAISS.load_local(
            path,
            retriever.embeddings,
            allow_dangerous_deserialization=True
        )
        return retriever

def main():
    # Initialize retriever
    knowledge_base_path = 'processed_jax_docs/jax_knowledge_base.json'
    index_path = "jax_faiss_index"
    
    print("Creating JAX FAISS retriever...")
    retriever = JAXFAISSRetriever(knowledge_base_path)
    
    print(f"Saving index to {index_path}...")
    retriever.save_index(index_path)
    
    print("Retriever created and index saved successfully!")
    return retriever

if __name__ == "__main__":
    retriever = main()