from dotenv import load_dotenv

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from typing import Sequence
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

# Load environment variables from .env
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    # repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    # max_new_tokens=512,
    # do_sample=False,
    # repetition_penalty=1.03,
)

chat_model = ChatHuggingFace(llm=llm)


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language} language.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

def call_model(state: State):
    prompt = prompt_template.invoke(state)
    response = chat_model.invoke(prompt)
    return {"messages": [response]}

# handle memory
workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc456"}}
query = "Hi! I'm Bob."
language = "Indonesia"

input_messages = [HumanMessage(query)]
output = app.invoke(
    {"messages": input_messages, "language": language},
    config,
)
output["messages"][-1].pretty_print()


########### WITH STREAMING

# config = {"configurable": {"thread_id": "abc789"}}
# query = "Hi I'm Risqi , Tell me about amazon web services"
# language = "English"

# input_messages = [HumanMessage(query)]
# for chunk, metadata in app.stream(
#     {"messages": input_messages, "language": language},
#     config,
#     stream_mode="messages",
# ):
#     if isinstance(chunk, AIMessage):  # Filter to just model responses
#         print(chunk.content, end="|")