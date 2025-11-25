from langchain.text_splitter import RecursiveCharacterTextSplitter,Language


# A multiline Markdown string that represents documentation text of a project
text = """
# Project Name: Smart Student Tracker

A simple Python-based project to manage and track student data, including their grades, age, and academic status.


## Features

- Add new students with relevant info
- View student details
- Check if a student is passing
- Easily extendable class-based design


##  Tech Stack

- Python 3.10+
- No external dependencies


## Getting Started

1. Clone the repo  
   ```bash
   git clone https://github.com/your-username/student-tracker.git

"""


# Initialize a Markdown-aware text splitter.
# It understands Markdown structure (headings, lists, code blocks)
# chunk_size=200 each chunk will contain a max of 200 characters
# chunk_overlap=0  chunks are fully independent (no repeated text between chunks)
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=200,
    chunk_overlap=0,
)

# Perform the split
chunks = splitter.split_text(text)

print(len(chunks)) # Print how many chunks were created
print(chunks[0]) # Print the first chunk to inspect the output