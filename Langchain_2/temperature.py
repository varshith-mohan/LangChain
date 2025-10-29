from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# gpt-4 is paid model

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=1.5)

result = model.invoke("Write a 5 line story on mahabharat")

print(result.content)