from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(repo_id='google/gemma-2-2b-it', task='text-generation', huggingfacehub_api_token='your api key')

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(template='Generate a detailed report on {topic}', input_variables=['topic'])

prompt2 = PromptTemplate(template='Generate a 5 pointer summary from the following text \n {text}', input_variables=['text'])

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':'Fitness'})

cleaned_text = result.replace('*', '')

print(cleaned_text)

chain.get_graph().print_ascii()