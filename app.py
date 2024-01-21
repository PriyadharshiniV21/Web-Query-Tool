import streamlit as st
from main import create_vector_db, get_qa_chain

st.title("Web Query Tool")
st.sidebar.title("Web URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placefolder = st.empty()

if process_url_clicked:
    create_vector_db(urls)

question = main_placefolder.text_input("Question: ")

if question:

    class Document:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata

    chain = get_qa_chain()
    response = chain(question)
    st.subheader("Answer: ")
    st.write(response["result"])

    sources = response.get("source_documents", "")

    if sources:
        st.subheader("Sources: ")
        sources_list = set([doc.metadata['source'] for doc in sources])
        for source in sources_list:
            st.write(source)