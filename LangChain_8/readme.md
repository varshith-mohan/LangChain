A vector store is a system designed to store and retrieve data represented as numerical vectors.

Key Features :-
1. Storage - Ensures that vectors and their associated metadata are retained, whether in-memory for quick lookups or on-disk for durability and large-scale use.
2. Similarity Search - Helps retrieve the vectors most similar to a query vector.
3. Indexing - Provide a data structure or method that enables fast similarity searches on high-dimensional vectors (e.g., approximate nearest neighbor lookups).
4. CRUD Operations - Manage the lifecycle of data—adding new vectors, reading them, updating existing entries, removing outdated vectors.

use cases -- 
- RAG
- Semantic Search
- Recommendation Systems
- Image/Multimedia search


Vector Store Vs Vector Database

Vector Store --
• Typically refers to a lightweight library or service that focuses on storing vectors (embeddings) and performing similarity search.
• May not include many traditional database features like transactions, rich query languages, or role-based access control.
• Ideal for prototyping, smaller-scale applications
• Examples: FAISS (where you store vectors and can query them by similarity, but you handle persistence and scaling separately).


Vector Database --
• A full-fledged database system designed to store and query vectors.
• Offers additional "database-like" features:
• Distributed architecture for horizontal scaling
• Durability and persistence (replication, backup/restore)
• Metadata handling (schemas, filters)
• Potential for ACID or near-ACID guarantees
• Authentication/authorization and more advanced security
• Geared for production environments with significant scaling, large datasets


Vector Stores in LangChain -- 

• Supported Stores: LangChain integrates with multiple vector stores (FAISS, Pinecone, Chroma, Qdrant, Weaviate, etc.), giving you flexibility in scale, features, and deployment.
• Common Interface: A uniform Vector Store API lets you swap out one backend (e.g., FAISS) for another (e.g., Pinecone) with minimal code changes.
• Metadata Handling: Most vector stores in LangChain allow you to attach metadata (e.g., timestamps, authors) to each document, enabling filter-based retrieval.


Chroma Vector Store -- 
Chroma is a lightweight, open-source vector database that is friendly for local development and small- to medium-scale production needs.