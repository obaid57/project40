import streamlit as st
import asyncio
import fitz  
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_ai21 import AI21Embeddings 

groq_api_key = "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"
ai21_api_key = "iKRWpmBGvRV6QiTDoZFq5gTCWv2dYJlm"

@st.cache_resource(show_spinner=False)
def load_pdf(file_path):
    text = ""
    pdf_document = fitz.open(file_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

@st.cache_resource(show_spinner=False)
def process_pdf(file_path):
    text_content = load_pdf(file_path)
    document = Document(page_content=text_content, metadata={"source": file_path})
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    final_documents = text_splitter.split_documents([document])
    
    embeddings = AI21Embeddings(api_key=ai21_api_key)  
    vectors = FAISS.from_documents(final_documents, embeddings)
    return vectors

if "vector" not in st.session_state:
    pdf_file_path = 'dx.pdf'  
    st.session_state.vectors = process_pdf(pdf_file_path)

st.title("DigitalEx ChatBot")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-70b-8192")

prompt_template = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
Do not add stuff. Make sure it is present in the document.
Provide the response in a step-by-step format if applicable.



<context>
{context}
<context>
Questions: {input}
"""
)

document_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Enter Your Question Here")

if prompt:
    response = retrieval_chain.invoke({"input": prompt})
    st.write(response['answer'])

    
    with st.expander("Document Similarity Search"):
        for doc in response["context"]:
            st.write(doc.page_content)
            st.write("--------------------------------")
