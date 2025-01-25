import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import re
from io import BytesIO
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Load Google API Key

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
def get_pdf_text(pdf_files):
    text = ""
    for uploaded_file in pdf_files:
        # Convert the uploaded file to a BytesIO stream
        pdf_stream = BytesIO(uploaded_file.read())
        pdf_reader = PdfReader(pdf_stream)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local('faiss_index')

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer.
    If there is 4 numerical columns in table, first 2 for months ended 31 march, next 2 for year ended march 31 
    If there is 2 numerical columns then those are march 31 2024, march 31 2023 in a row. These are for numerical columns only.
    
    Context:\n{context}?\n
    Question:\n{question}\n
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model='gemini-1.5-flash',temparature=0.3)
    prompt = PromptTemplate(template=prompt_template,input_variables=["context",'question'])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    # Ensure embeddings match the creation process
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

    # Load the FAISS index with dangerous deserialization explicitly allowed
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=3)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, 'question': user_question},
        return_only_outputs=True)
    #print(response)
    st.markdown("## Answer: ")
    st.markdown(response["output_text"])
    st.markdown("## Supporting Documents: ")
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0.5)
    response1 = model.invoke(f"Extract entire tables from {docs} which is use for get answer for question {user_input} as {response['output_text']}. Don't give any information other than the tables and small parts of the table. Give total table")
    st.markdown(response1.content)
    #st.markdown("="*88)
def main():
    st.header('Financial Document QA System')

    user_question = st.text_input("Ask a Question")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu: ")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True, type=["pdf"])
        if pdf_docs and st.button("Submit & Process"):
            with st.spinner("Processing..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("PDF Text Processing Successful!")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        st.sidebar.markdown("""
        ### How to use this application:
        1. Upload your financial PDF document (preferably containing P&L statements and financial tables)
        2. Ask questions about the financial data in natural language
        3. View the answer and supporting information from the document
        
        #### Example questions:
        - "What was the total revenue for Q1 2024?"
        - "How do operating expenses compare between Q3 and Q4?"
        - "What is the trend in gross profit margin over the past year?"
        """)

if __name__ == "__main__":
    main()