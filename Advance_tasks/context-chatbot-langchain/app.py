import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
import os

Gemini_api_key = os.getenv("GEMINI_API_KEY")


loader = TextLoader("your_doc.txt")
docs = loader.load()


embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)


qa = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0),
    retriever=db.as_retriever(),
)

# UI
st.title("🧠 Context-Aware Chatbot")
chat_history = []

query = st.text_input("Ask something:")
if query:
    result = qa({"question": query, "chat_history": chat_history})
    chat_history.append((query, result['answer']))
    st.write(result['answer'])
