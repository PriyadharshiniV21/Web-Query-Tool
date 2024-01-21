import os
from langchain.chat_models import google_palm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from dotenv import load_dotenv

# Take environment variables from .env.
load_dotenv()

# Create vector database
filepath = "faiss_vectorIndex"

# Create Google Palm LLM model
llm = google_palm.ChatGooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"])

# Create embeddings
embeddings = GooglePalmEmbeddings()

def create_vector_db(urls):

    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(data)

    # Save the embeddings to FAISS index
    vector_index = FAISS.from_documents(docs, embeddings)

    # Save vector database locally
    vector_index.save_local(filepath)

def get_qa_chain():
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide simple answers only from the text extracted from the provided urls which is in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}

    if os.path.exists(filepath):
        vectorIndex = FAISS.load_local(filepath, embeddings)

        # Create a retriever for querying the vector database
        retriever = vectorIndex.as_retriever(score_threshold=0.7)

        chain = RetrievalQA.from_chain_type(llm=llm,
                                            chain_type="stuff",
                                            retriever=retriever,
                                            input_key="query",
                                            return_source_documents=True,
                                            chain_type_kwargs=chain_type_kwargs)

        return chain
