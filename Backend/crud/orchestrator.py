from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from fastapi import APIRouter
from dotenv import load_dotenv

load_dotenv()



test = APIRouter()

model = ChatOpenAI(model = 'gpt-4o')
parser = JsonOutputParser()




global chat_history
chat_history = []

@test.post("/query")
async def query_endpoint(query : str):
    chain = prompt | model | parser
    result = chain.invoke({ "chat_history": chat_history, "question": query})
    return result
