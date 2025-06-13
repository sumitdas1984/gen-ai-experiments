import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import time


load_dotenv()

## load the GROQ And OpenAI API KEY 
groq_api_key=os.getenv('GROQ_API_KEY')


llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions based on the provided context."
              "If you don't know the answer, say 'I don't know'.\n\n"
              "Context:\n{context}"),
    ("human", "{input}"),
])

def vector_embedding():

    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'} # Use 'cuda' if you have a GPU
        )
        st.session_state.loader=PyPDFDirectoryLoader("./documents") ## Data Ingestion
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs) #splitting
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector OpenAI embeddings


def main():
    # Set page title
    st.set_page_config(page_title="Document Q&A", layout="wide")
    
    # Add title
    st.title("Document Q&A")
    
    # Add text input for document content
    prompt1 = st.text_area("Enter your question from document",)
    
    # Add Generate Embeddings button
    if st.button("Generate Embeddings"):
        vector_embedding()        
        st.write("Vector Store DB Is Ready")

    if prompt1:
        document_chain=create_stuff_documents_chain(llm,prompt)
        retriever=st.session_state.vectors.as_retriever()
        retrieval_chain=create_retrieval_chain(retriever,document_chain)
        start=time.process_time()
        response=retrieval_chain.invoke({'input':prompt1})
        print("Response time :",time.process_time()-start)
        st.write(response['answer'])

        # With a streamlit expander
        with st.expander("Document Similarity Search"):
            # Find the relevant chunks
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")


if __name__ == "__main__":
    main()
