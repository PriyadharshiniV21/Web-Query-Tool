import os
import streamlit as st
import pickle
import time
import langchain
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from apikey import APIKEY

os.environ["OPENAI_API_KEY"] = APIKEY

# Initialize LLM with required params
llm = OpenAI(temperature=0.9, max_tokens=500)

# Load data
loader = UnstructuredURLLoader(urls=[
    "https://economictimes.indiatimes.com/markets/stocks/live-blog/bse-sensex-today-live-nifty-stock-market-updates-22-november-2023/liveblog/105399260.cms",
    "https://www.moneycontrol.com/news/photos/business/stocks/gainers-and-losers-10-stocks-that-moved-the-most-on-november-22-11791681.html"])

data = loader.load()

# Split data to create chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = text_splitter.split_documents(data)

# Create embeddings for these chunks and save them to FAISS index
embeddings = OpenAIEmbeddings()

vector_index = FAISS.from_documents(docs, embeddings)

# file_path = "vector_index.pkl"
# with open(file_path, "wb") as f:
    # pickle.dump(vector_index, f)

# Retrieve similar embeddings for a given question and call LLM to retrieve final answer
chain = RetrievalQAWithSourcesChain.from_llm(llm, retriever=vector_index.as_retriever())

query = "What is the CMP of Ugro Capital?"

langchain.debug = True
chain({"question": query}, return_only_outputs=True)