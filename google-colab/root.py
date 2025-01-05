from langchain_aws import BedrockLLM
from langchain_aws import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.schema.runnable import RunnableLambda
from dotenv import load_dotenv
import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# Load environment variables from .env
load_dotenv()

bedrockmodel = ChatBedrock(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    streaming=True,
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_ID"),
    region=os.environ.get("AWS_REGION_BEDROCK"),
)