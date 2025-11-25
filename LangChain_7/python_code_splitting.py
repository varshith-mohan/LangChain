from langchain.text_splitter import RecursiveCharacterTextSplitter,Language

text = """
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade  # Grade is a float (like 8.5 or 9.2)

    def get_details(self):
        return self.name"

    def is_passing(self):
        return self.grade >= 6.0


# Example usage
student1 = Student("Aarav", 20, 8.2)
print(student1.get_details())

if student1.is_passing():
    print("The student is passing.")
else:
    print("The student is not passing.")

"""

# Initialize a Python-aware RecursiveCharacterTextSplitter.
# language=Language.PYTHON Splits text using Python syntax rules
# chunk_size=300 Max characters per chunk
# chunk_overlap=0 No repeated text between chunks
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0,
)

# Perform the split
chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[1])

# 1. A Python code snippet is stored in a multi-line string.
# 2. A RecursiveCharacterTextSplitter is created with Python-specific parsing.
# 3. The text is split into chunks of up to 300 characters.
# 4. No overlap means every chunk contains unique text.
# 5. Finally, the total number of chunks and the second chunk is printed.