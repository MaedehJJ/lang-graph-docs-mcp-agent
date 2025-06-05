import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore


@tool
def langgraph_query_tool(query: str):
    """
    Query the LangGraph documentation using a retriever.

    Args:
        query (str): The query to search the documentation with

    Returns:
        str: A str of the retrieved documents
    """
    retriever = SKLearnVectorStore(
        embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
        persist_path=os.getcwd() + "/sklearn_vectorstore.parquet",
        serializer="parquet",
    ).as_retriever(search_kwargs={"k": 3})

    relevant_docs = retriever.invoke(query)
    print(f"Retrieved {len(relevant_docs)} relevant documents")
    formatted_context = "\n\n".join(
        [
            f"==DOCUMENT {i+1}==\n{doc.page_content}"
            for i, doc in enumerate(relevant_docs)
        ]
    )
    return formatted_context


load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
augmented_llm = llm.bind_tools([langgraph_query_tool])

instructions = """You are a helpful assistant that can answer questions about the LangGraph documentation. 
Use the langgraph_query_tool for any questions about the documentation.
If you don't know the answer, say "I don't know."""

messages = [
    {"role": "system", "content": instructions},
    {"role": "user", "content": "What is LangGraph?"},
]

message = augmented_llm.invoke(messages)
message.pretty_print()
