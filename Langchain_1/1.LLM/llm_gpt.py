from langchain_openai import OpenAI
from dotenv import load_dotenv # loads api key from .env


load_dotenv()

llm = OpenAI(model = 'gpt-3.5-turbo')

result = llm.invoke('What is the capital of UK')

print(result)