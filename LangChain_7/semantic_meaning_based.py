from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
# Import SemanticChunker for meaning-based text splitting
# and OpenAIEmbeddings to generate embeddings used for semantic segmentation.

from dotenv import load_dotenv

load_dotenv() # Import load_dotenv to load environment variables (eg OPENAI_API_KEY)

# Initialize SemanticChunker:
#  Uses embeddings to split text based on meaning, not characters.
#  breakpoint_threshold_type="standard_deviation": Uses standard deviation of embedding distances to detect boundaries.
#  breakpoint_threshold_amount=3: Controls how sensitive the chunking should be (higher = fewer chunks).
text_splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=3
)

# The goal is for SemanticChunker to group sentences by meaning.
sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.


Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""

# Convert the text into semantic chunks.
# The chunking is driven by meaning (embeddings), NOT by character length.
# Sentences about similar concepts will form the same chunk.
docs = text_splitter.create_documents([sample])
print(len(docs))
print(docs)

# 1. Load environment variables (including OpenAI API key).
# 2. Initialize SemanticChunker with OpenAIEmbeddings for meaning-based splitting.
# 3. Provide a text sample containing 3 different topics:
#       - Farming
#       - Cricket (IPL)
#       - Terrorism and safety
# 4. SemanticChunker analyzes embedding similarity between sentences.
# 5. Boundaries between unrelated topics are detected using standard deviation rules.
# 6. The text is split into meaningfully grouped chunks.
# 7. Final chunks (as Document objects) are printed.
