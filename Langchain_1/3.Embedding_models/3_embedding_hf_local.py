from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')


text = 'Langchain is goldmine'

documents = [
     'New Delhi is capital of India',
    'Paris is capital of France'
]

vector1 = embedding.embed_query(text)
vector2 = embedding.embed_documents(documents)

print(str(vector1))
print('\n')
print(str(vector2))