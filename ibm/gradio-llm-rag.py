from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
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

def process_query(file, query, use_rag):
    """Process a query using either RAG or direct LLM."""
    if use_rag:
        if not file:
            return "Please provide a PDF file for RAG mode."
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
            return f"An error occurred in RAG mode: {str(e)}"
    else:
        try:
            # Direct LLM query without RAG
            response = llm.invoke(query)
            return response.content
        except Exception as e:
            return f"An error occurred in direct LLM mode: {str(e)}"

def update_interface(use_rag):
    """Update the interface based on RAG mode selection."""
    if use_rag:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

# Create Gradio interface
with gr.Blocks() as rag_application:
    gr.Markdown("# RAG/LLM Chatbot")
    gr.Markdown("Toggle between RAG and direct LLM modes to interact with the model.")
    
    with gr.Row():
        use_rag = gr.Checkbox(label="Use RAG Mode", value=True)
    
    with gr.Row():
        pdf_file = gr.File(
            label="Upload PDF File",
            file_count="single",
            file_types=['.pdf'],
            type="filepath",
            visible=True
        )
    
    with gr.Row():
        query_input = gr.Textbox(
            label="Input Query",
            lines=2,
            placeholder="Type your question here..."
        )
    
    with gr.Row():
        submit_btn = gr.Button("Submit")
    
    output = gr.Textbox(label="Output")
    
    # Event handlers
    use_rag.change(fn=update_interface, inputs=[use_rag], outputs=[pdf_file])
    submit_btn.click(
        fn=process_query,
        inputs=[pdf_file, query_input, use_rag],
        outputs=[output]
    )

if __name__ == "__main__":
    rag_application.launch(server_name="0.0.0.0", server_port=7860)