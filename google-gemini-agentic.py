#  refer readme.txt to run the project

# This code include, Sample Agent Code (Integration)

import pandas as pd

from singlestoredb.server import docker
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_singlestore import SingleStoreVectorStore

# New imports for Google Gemini
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

########################################
#
#   Agent libs
#
########################################

from langchain.tools import Tool

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

########################################
#
########################################

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

        
        ########################################
        #
        #   Agent intergration: start
        #
        ########################################

        # Define prompt template with clean formatting
        # template = """
        #             You are an expert in answering questions about a pizza restaurant.

        #             Here are some relevant reviews: {reviews}

        #             Here is the question: {question}
        #             """
        
        # Define a custom function wrapper for the retriever
        def get_pizza_context(query: str):
            """Tool to retrieve relevant pizza reviews from the SingleStore database."""
            # The retriever returns documents, so we join their content for the LLM
            docs = retriever.invoke(query)
            return "\n---\n".join([doc.page_content for doc in docs])
        
        # Convert the function into a tool
        pizza_tool = Tool(
            name="pizza_review_search",
            func=get_pizza_context,
            description="A tool for searching and retrieving specific pizza reviews."
        )


        # The prompt guides the Agent to use the tool
        agent_prompt = PromptTemplate.from_template(
                """
                You are an expert agent that answers questions about a pizza restaurant. 
                You have access to the following tools:
                {tools}

                You must use a tool before answering any question about reviews. 
                The only tool name available is: {tool_names}

                The user's question is: {input}

                {agent_scratchpad}
                """
        )
        # Create the Agent logic
        agent = create_react_agent(
                llm=model, # Your ChatGoogleGenerativeAI model
                tools=[pizza_tool],
                prompt=agent_prompt, # Now includes all required variables
        )
        # Create the Agent Executor (the runnable system)
        # Create the Agent Executor with the parsing error handler (FIX)
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=[pizza_tool], 
            verbose=True, 
            handle_parsing_errors=True # <-- FIX 1: Handles LLM formatting mistakes
        )
        #prompt = ChatPromptTemplate.from_template(template)
        #chain = prompt | model

        ########################################
        #
        #   Agent intergration: end
        #
        ########################################

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
            
            ########################################
            #
            #   Agent invocation
            #
            ########################################

            #reviews = retriever.invoke(user_input)
            #result = chain.invoke({"reviews": reviews, "question": user_input})
            
            # The agent handles retrieval internally
            result = agent_executor.invoke({"input": user_input})


            print("\n--- Answer ---")
            print(result)



if __name__ == "__main__":
    # Ensure your GOOGLE_API_KEY environment variable is set
    main()