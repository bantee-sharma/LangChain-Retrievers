from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# Step 1: Your source documents
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

embeddings = HuggingFaceEmbeddings()

db = FAISS.from_documents(
    documents=docs,
    embedding=embeddings,
    
)

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs = {"k":3, "lambda_mult":0.5}
)

query = "What is langchain?"
res = retriever.invoke(query)
print(res)

