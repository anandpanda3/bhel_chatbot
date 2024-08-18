import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables and configure API
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
print(f"API Key loaded: {'Yes' if api_key else 'No'}")  # Debug print
genai.configure(api_key=api_key)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        print("Vector store saved successfully")  # Debug print
        return True
    except Exception as e:
        print(f"Error in get_vector_store: {str(e)}")
        st.error(f"An error occurred while processing the document: {str(e)}")
        return False

def get_conversational_chain():
    prompt_template = """
    Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question, k=4)  # Retrieve top 4 most relevant chunks
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        print(f"Response: {response}")  # Debug print
        st.write("Reply: ", response["output_text"])
    except Exception as e:
        print(f"Error in user_input: {str(e)}")
        st.error(f"An error occurred while processing your question: {str(e)}")

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")
    print("Main function started")  # Debug print

    st.warning("This application uses local file deserialization. Only use this with files you trust and have created yourself.")

    # Use session state to keep track of whether PDFs have been processed
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False

    if st.session_state.pdf_processed:
        st.info("PDF data has been processed. You can ask questions now.")
        user_question = st.text_input("Ask a Question from the PDF Files")
        print(f"User question: {user_question}")  # Debug print

        if user_question:
            user_input(user_question)
    else:
        st.info("Please upload and process PDF files before asking questions.")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            print("Submit & Process button clicked")  # Debug print
            if pdf_docs:
                with st.spinner("Processing..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        success = get_vector_store(text_chunks)
                        if success:
                            st.session_state.pdf_processed = True
                            st.success(f"Done! Processed {len(pdf_docs)} PDF(s). You can now ask questions.")
                            st.rerun()
                        else:
                            st.error("Failed to process PDFs. Please try again.")
                    except Exception as e:
                        print(f"Error in processing: {str(e)}")
                        st.error(f"An error occurred during processing: {str(e)}")
            else:
                st.warning("Please upload PDF files before processing.")
        
        if st.button("Clear All"):
            if os.path.exists("faiss_index"):
                os.remove("faiss_index")
            st.session_state.pdf_processed = False
            st.success("All data cleared. You can upload new PDFs now.")
            st.rerun()

if __name__ == "__main__":
    main()