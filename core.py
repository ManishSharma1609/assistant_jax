import os
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import JSONLoader
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()

# Load the FAISS Index
INDEX_PATH = "jax_faiss_index"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
docsearch = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# Define the LLM
chat = ChatOpenAI(model="gpt-4", temperature=0)

# Define the JAX-specific prompt template
jax_prompt_template = """You are an expert in JAX and machine learning. Use the provided context to answer the user's query.
If the query involves code, always provide a complete and executable JAX example.

Context:
-------------------
{context}

User Query: {input}

When providing code examples:
- Use `jax.numpy as jnp` for NumPy-like operations.
- Ensure the code is correctly formatted in Markdown.
- Include type annotations where relevant.
- Add comments for clarity.

Response format:
[Explanation] A brief explanation of the concept.
[Code Example] (if applicable):
```python
# JAX code here
```
"""

jax_prompt = PromptTemplate(template=jax_prompt_template, input_variables=["context", "question"])

# Create document retrieval and query handling
stuff_document_chain = create_stuff_documents_chain(chat, jax_prompt)

# Create a history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm=chat, retriever=docsearch.as_retriever(), prompt=jax_prompt
)

qa_chain = create_retrieval_chain(
    retriever=history_aware_retriever, combine_docs_chain=stuff_document_chain
)

def query_jax_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    """Queries the JAX LLM assistant with context-aware retrieval."""
    result = qa_chain.invoke(input={"input": query, "chat_history": chat_history})
    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"],
    }
    return new_result

if __name__ == "__main__":
    res = query_jax_llm(query="How to use jax.vmap with multiple arguments?")
    print(res["result"])
