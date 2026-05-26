from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama

# Load docs
loader = DirectoryLoader("langchain-docs")
docs = loader.load()

# Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)
chunks = splitter.split_documents(docs)

# Embeddings
embeddings = OllamaEmbeddings(model="llama3")

# Vector DB

# vectorstore = FAISS.from_documents(chunks, embeddings)
# retriever = vectorstore.as_retriever()

# Vector DB - instead of retrieval lets persist 
import os
if os.path.exists("faiss_index"):
    print("loading existing FAISS Index ....")
    vectorstore=FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    print("creating faiss index ...")
    vectorstore=FAISS.from_documents(chunks,embeddings)
    vectorstore.save_local("faiss_index")
retriever = vectorstore.as_retriever() 

# LLM
llm = ChatOllama(model="llama3", temperature=0)

# Simple RAG loop (NO RetrievalQA)
print("\nRAG Chatbot Ready\n")

while True:
    query = input("Ask: ")
    if query.lower() in ["exit", "quit"]:
        break

    docs = retriever.invoke(query)

    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
    Answer using only this context:

    {context}

    Question: {query}
    """

    response = llm.invoke(prompt)

    print("\nAnswer:", response.content)