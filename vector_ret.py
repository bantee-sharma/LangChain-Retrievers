from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# Step 1: Your source documents
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

embeddings = HuggingFaceEmbeddings()

db = FAISS.from_documents(
    documents=documents,
    embedding=embeddings,
    
)

retriever = db.as_retriever(search_kwargs={"k":2})

query = "What is use of chroma?"
res = retriever.invoke(query)
print(res)

