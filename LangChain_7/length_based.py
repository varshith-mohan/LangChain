from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('Devops.pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=''
)

result = splitter.split_documents(docs)

print(result[1].page_content)


# Loads a PDF file
# PyPDFLoader reads the file Devops.pdf and extracts all pages as documents.
# Stores all PDF pages into a list
# docs = loader.load() returns a list where each element represents one page of the PDF.
# Creates a text splitter
# CharacterTextSplitter is configured to:
# Split text into chunks of 200 characters
# No overlap between chunks
# Use an empty string ('') as the separator -> splits text purely based on size
# Splits the PDF into text chunks
# splitter.split_documents(docs) breaks each page's text into 200-character chunks and returns them.
# Prints the content of the second chunk
# result[1].page_content prints the text of chunk #2 from the split results.