from langchain_aws import BedrockLLM
from langchain_aws import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.schema.runnable import RunnableLambda
from dotenv import load_dotenv
import os
from root import bedrockmodel

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# Load environment variables from .env
load_dotenv()


# llm = HuggingFaceEndpoint(
#     repo_id="HuggingFaceH4/zephyr-7b-beta",
#     task="text-generation",
#     max_new_tokens=512,
#     do_sample=False,
#     repetition_penalty=1.03,
# )

# chat_model = ChatHuggingFace(llm=llm)

# result = bedrockmodel.invoke("Hello, my name is Bob")
# print(result.content)

# messages = [
#     ("system", "You are a comedian who tells jokes about {topic}."),
#     ("human", "Tell me {joke_count} jokes."),
# ]

# ptr = ChatPromptTemplate.from_messages(messages)
# chain = ptr | bedrockmodel | StrOutputParser()
# result = chain.invoke({"topic": "lawyers", "joke_count": 3})
# print(result)


# Create messages for the prompt template
tmp = """
You are a helpful assistant that will do translation.
Translate this sentences in {lang} language.
Context:
{msg}
Answer:
"""

# Create prompt template
prompt = ChatPromptTemplate.from_template(tmp)

uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")

# Create the chain
chain = prompt | bedrockmodel | StrOutputParser() | uppercase_output | count_words

# Invoke the chain
result = chain.invoke({
    "lang": "Indonesian",  # Note: "Indonesia" is the country, "Indonesian" is the language
    "msg": "Hello there"
})

print(result)