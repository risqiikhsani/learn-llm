from langchain_aws import ChatBedrock
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import gradio as gr
import requests
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from datetime import datetime
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage,AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import os
load_dotenv()
search = TavilySearchResults(max_results=5)
search_results = search.invoke("what is the weather in SF")
print(search_results)

def initialize_model():
    # model = HuggingFaceEndpoint(
    #     repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    # )
    # return ChatHuggingFace(llm=model)
    bedrockmodel = ChatBedrock(
        model="anthropic.claude-3-haiku-20240307-v1:0",
        streaming=True,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_ID"),
        region=os.environ.get("AWS_REGION_BEDROCK"),
    )
    return bedrockmodel

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

# If we want, we can create other tools.
# Once we have all the tools we want, we can put them in a list that we will reference later.
tools = [search]
# create agent
# Create agent with increased recursion limit and better system prompt
agent_executor = create_react_agent(
    model=llm,
    tools=tools,
)

def process_query(file, query, use_rag, use_search_tool):
    """Process a query using RAG, direct LLM, or search tool."""
    try:
        if use_search_tool:
            # Properly format the query for the agent
            messages = [
                HumanMessage(content=f"Please help me with this question: {query}")
            ]
            response = agent_executor.invoke({"messages": messages},{
                "recursion_limit": 100,
            })
            # Extract the last AI message content
            for msg in reversed(response["messages"]):
                if isinstance(msg, AIMessage):
                    # return msg.content
                    return msg.content[0]['text']
            return "No response generated"
        elif use_rag:
            if not file:
                return "Please provide a PDF file for RAG mode."
            retriever_obj = retriever(file)
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever_obj,
                return_source_documents=False
            )
            response = qa.invoke(query)
            return response['result']
        else:
            # Direct LLM query
            response = llm.invoke(query)
            return response.content
    except Exception as e:
        return f"An error occurred: {str(e)}"

def update_interface(use_rag, use_search_tool):
    """Update the interface based on mode selection."""
    if use_search_tool:
        return gr.update(visible=False), gr.update(placeholder="Ask about the search (e.g., 'What's the weather in London?')")
    elif use_rag:
        return gr.update(visible=True), gr.update(placeholder="Type your question about the document here...")
    else:
        return gr.update(visible=False), gr.update(placeholder="Ask anything...")

# Create Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# AI Assistant with RAG, LLM, and Search Tool")
    gr.Markdown("Choose your mode and interact with the assistant.")
    
    with gr.Row():
        with gr.Column(scale=1):
            use_rag = gr.Checkbox(label="Use RAG Mode", value=True)
        with gr.Column(scale=1):
            use_search_tool = gr.Checkbox(label="Use Search Tool", value=False)
    
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
    use_rag.change(
        fn=update_interface,
        inputs=[use_rag, use_search_tool],
        outputs=[pdf_file, query_input]
    )
    use_search_tool.change(
        fn=update_interface,
        inputs=[use_rag, use_search_tool],
        outputs=[pdf_file, query_input]
    )
    submit_btn.click(
        fn=process_query,
        inputs=[pdf_file, query_input, use_rag, use_search_tool],
        outputs=[output]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)