#  refer readme.txt to run the project

import pandas as pd
from singlestoredb.server import docker
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_singlestore import SingleStoreVectorStore

# New imports for Google Gemini
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI


def setup_database(s2db):
    """Initialize the SingleStore database."""
    with s2db.connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute("CREATE DATABASE IF NOT EXISTS testdb")


def load_documents():
    """Load pizza reviews from CSV and convert to Document objects."""
    # Ensure pizza_reviews.csv has all fields quoted if they contain commas!
    df = pd.read_csv("pizza_reviews.csv") 
    documents = []
    for i, row in df.iterrows():
        content = f"{row['Title']} {row['Review']}"
        documents.append(
            Document(
                page_content=content,
                metadata={"rating": row["Rating"], "date": row["Date"]},
                id=str(i)
            )
        )
    return documents


def main():
    """
    Run a pizza review Q&A application using SingleStoreDB vector store and 
    Google Gemini.
    """
    print("Starting SingleStoreDB server for vector storage...")
    # NOTE: Ensure Docker Desktop is running!
    with docker.start(license="") as s2db:
        setup_database(s2db)

        print("Loading and embedding pizza reviews...")
        documents = load_documents()
        
        # 1. Initialize Embeddings Model (Gemini fix)
        embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

        # 2. Set up vector store using the CONSTRUCTOR (Fix for TypeErrors)
        vector_store = SingleStoreVectorStore(
            embedding=embedding,
            host=s2db.connection_url, # Using the host= keyword argument
            database="testdb",
            table_name="pizza_reviews",
        )
        # 3. Add documents after the store is initialized
        vector_store.add_documents(documents)

        # Create retriever that fetches the 2 most relevant reviews for each query
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})

        print("Initializing Gemini LLM...")
        # 4. Initialize LLM (Gemini fix)
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

        # Define prompt template with clean formatting
        template = """
You are an expert in answering questions about a pizza restaurant.

Here are some relevant reviews: {reviews}

Here is the question: {question}
"""
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model

        print("\n------------------------------------------")
        print("Pizza Review Question & Answer System")
        print("Ask questions about pizza reviews, and the system will find relevant reviews")
        print("and generate an answer based on those reviews.")
        print("------------------------------------------\n")

        while True:
            user_input = input("\nEnter your question about pizza (or 'exit' to quit): ")
            if user_input.lower() == "exit":
                break
            print("\nFinding relevant reviews and generating answer...")
            reviews = retriever.invoke(user_input)
            result = chain.invoke({"reviews": reviews, "question": user_input})

            print("\n--- Answer ---")
            print(result)


if __name__ == "__main__":
    # Ensure your GOOGLE_API_KEY environment variable is set
    main()