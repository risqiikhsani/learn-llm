from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Corrected import
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
)  # Corrected import for embeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers.string import StrOutputParser
# Specify the file path
file_path = "./data.csv"

try:
    # Load the CSV file
    loader = CSVLoader(file_path=file_path)
    data = loader.load()
    print("First document content:")
    print(data[0].page_content)  # Print the first document's content
except FileNotFoundError:
    print(
        f"Error: File '{file_path}' not found. Please check the file path and try again."
    )
    exit(1)

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_documents(data)
# print(f"Number of text chunks: {len(texts)}")
# print(texts[:3])  # Print the first three chunks for verification

# Initialize the HuggingFace embeddings model
# Specify the correct Hugging Face model endpoint if required
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)  # Example model name

# Generate embeddings for the text chunks
# embeddings_result = embeddings_model.embed_documents([text.page_content for text in texts])
# print(f"Number of embeddings: {len(embeddings_result)}")

# embed query
# query = "What is the highest Socioeconomic score? "
# embedding_query_result = embeddings_model.embed_query(query)
# print(len(embedding_query_result))



vector_store = InMemoryVectorStore(embeddings_model)
for text in texts:
    vector_store.add_documents([text])

# query = "What is highest Socio score ?"
# docs = vector_store.similarity_search(query)
# print(len(docs))

model = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

llm = ChatHuggingFace(llm=model)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Define a system prompt that tells the model how to use the retrieved context
system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Context: {context}:"""
# Define a question
question = """What is the highest Socioeconomic score?"""

# Retrieve relevant documents
docs = retriever.invoke(question)

# Combine the documents into a single string
docs_text = "".join(d.page_content for d in docs)

# Populate the system prompt with the retrieved context
system_prompt_fmt = system_prompt.format(context=docs_text)

messages = [SystemMessage(content=system_prompt_fmt), HumanMessage(content=question)]



chain = llm | StrOutputParser()

response = chain.invoke(messages)

print(response)
