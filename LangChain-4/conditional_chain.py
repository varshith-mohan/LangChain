from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

llm2 = HuggingFaceEndpoint(repo_id='meta-llama/Llama-3.1-8B-Instruct', task='text-generation', huggingfacehub_api_token='your api key')

model2 = ChatHuggingFace(llm=llm2)

parser = StrOutputParser()


class Feedback(BaseModel):

    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)



prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)




classifier_chain = prompt1 | model2 | parser2


prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback'])


prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback'])


# cleaner function to remove Markdown stars
def clean_output(text: str) -> str:
    return text.replace('*', '').strip()

# conditional branching based on sentiment
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model2 | parser | RunnableLambda(clean_output)),
    (lambda x: x.sentiment == 'negative', prompt3 | model2 | parser | RunnableLambda(clean_output)),
    RunnableLambda(lambda x: 'could not find the suitable sentiment'),
)

# combine both chains
chain = classifier_chain | branch_chain

# examples
print('\n Positive Example:\n')
print(chain.invoke({'feedback': 'This is a beautiful waterfall'}))

print('\n Negative Example:\n')
print(chain.invoke({'feedback': 'The pizza was terrible'}))

print('\n Neutral Example:\n')
print(chain.invoke({'feedback': 'The weather is fine'}))

chain.get_graph().print_ascii()