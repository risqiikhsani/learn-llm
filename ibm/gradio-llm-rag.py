# !pip install langchain
# !pip install langchain-chroma
# !pip install transformers==4.46.0 tokenizers==0.20.3
# !pip install pypdf
# !pip install langchain-community
# !pip install langchain-huggingface
# !pip install gradio

from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import gradio as gr

def initialize_model():
    model = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        task="text-generation",
    )
    return ChatHuggingFace(llm=model)

def document_loader(file):
    """Load and parse a PDF file."""
    loader = PyPDFLoader(file.name)
    return loader.load()

def text_splitter(data):
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )
    return splitter.split_documents(data)

def vector_database(chunks):
    """Create a vector database from document chunks."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={'normalize_embeddings': False}
    )
    return Chroma.from_documents(chunks, embeddings)

def retriever(file):
    """Create a retriever from a PDF file."""
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    return vectordb.as_retriever()



# Initialize the model
llm = initialize_model()

def retriever_qa(file, query):
    """Process a query using RAG."""
    if not file or not query:
        return "Please provide both a PDF file and a query."
    
    try:
        retriever_obj = retriever(file)
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever_obj,
            return_source_documents=False
        )
        response = qa.invoke(query)
        return response['result']
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Create Gradio interface
rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(
            label="Upload PDF File",
            file_count="single",
            file_types=['.pdf'],
            type="filepath"
        ),
        gr.Textbox(
            label="Input Query",
            lines=2,
            placeholder="Type your question here..."
        )
    ],
    outputs=gr.Textbox(label="Output"),
    title="RAG Chatbot",
    description="Upload a PDF document and ask any question. The chatbot will answer using the provided document."
)

if __name__ == "__main__":
    rag_application.launch(server_name="0.0.0.0", server_port=7860)