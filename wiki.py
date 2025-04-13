from langchain_community.retrievers import WikipediaRetriever

query = "Geopolitical history of india and pakistan from the prespective of chinese."

retriever = WikipediaRetriever(top_k_results=2,lang="en")

docs = retriever.invoke(query)

for i, doc in enumerate(docs):
    print(f"\n--- Document {i+1} ---")
    print(doc.page_content[:1000])  # Print only first 1000 chars
    print(f"\nSource: {doc.metadata.get('title')}")