import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from models import check_if_model_is_available
from document_loader import load_documents
import argparse
import sys
import os
import time
import ollama
from typing import Dict, Generator


TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
DEFAULT_MODEL = "dolphin-mistral"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_PATH = "reports"
PERSIST_DIR = "db"
PROMPT_TEMPLATE = """
## Instruction:
Act as a helpful assistant, answering the question based solely on the provided context. Do not use any inherent knowledge that goes beyond the given context. Ensure the response is concise, clear, and directly addresses the question.

## Context:
{context}

## Question:
{question}

## Answer:
"""
PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)

def load_documents_into_database(model_name: str, documents_path: str) -> Chroma:
    """
    Loads documents from the specified directory into the Chroma database
    after splitting the text into chunks.

    Args:
        model_name (str): The name of the model to use for generating embeddings.
        documents_path (str): The path to the directory containing the documents.

    Returns:
        Chroma: The Chroma database with loaded documents.
    """

    print("Loading documents")
    raw_documents = load_documents(documents_path)
    if not raw_documents:
        if os.path.exists(PERSIST_DIR):
            db = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=OllamaEmbeddings(model=model_name)
            )
            return db
        else:
            raise FileNotFoundError("No documents found in the specified directory")
        
    documents = TEXT_SPLITTER.split_documents(raw_documents)

    print("Creating embeddings and loading documents into Chroma")
    start = time.time()
    db = Chroma.from_documents(
        documents,
        OllamaEmbeddings(model=model_name),
        persist_directory=PERSIST_DIR,
    )
    end_time = time.time() - start
    print(f"Time to load documents into Chroma: {end_time:.2f} seconds")
    db.persist()
    return db

def ollama_generator(model_name: str, messages: Dict) -> Generator:
    stream = ollama.chat(model=model_name, messages=messages, stream=True)
    for chunk in stream:
        yield chunk["message"]["content"]

def main(llm_model_name: str, embedding_model_name: str, documents_path: str, nb_docs: int) -> None:
    # Check to see if the models available, if not attempt to pull them
    try:
        check_if_model_is_available(llm_model_name)
        check_if_model_is_available(embedding_model_name)
    except Exception as e:
        print(e)
        sys.exit()

    try:
        db = load_documents_into_database(embedding_model_name, documents_path)
    except FileNotFoundError as e:
        print(e)
        sys.exit()

    st.image('images/logo.webp', use_column_width=True)
    st.title("CTrag")
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "dolphin-mistral:latest"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []
    st.session_state.selected_model = st.selectbox(
        "Please select the model:", [model["name"] for model in ollama.list()["models"]],
        index=[model["name"] for model in ollama.list()["models"]].index(st.session_state.selected_model)
    )
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("What is your cyber threat intelligence question?"):
        # Add user message to chat history
        docs = db.similarity_search(prompt, k=nb_docs)
        with st.expander("View Context"):
            st.write(docs)
        formated_prompt = PROMPT_TEMPLATE.format(context=docs, question=prompt)
        st.session_state.rag_messages.append({"role": "user", "content": formated_prompt})
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = st.write_stream(
                ollama_generator(
                    st.session_state.selected_model, st.session_state.rag_messages
                )
            )
        st.session_state.rag_messages.append({"role": "assistant", "content": response})
        st.session_state.messages.append({"role": "assistant", "content": response})


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local LLM with RAG with Ollama.")
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help=f"The name of the LLM model to use. Default is {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "-e",
        "--embedding_model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"The name of the embedding model to use. Default is {DEFAULT_EMBEDDING_MODEL}.",
    )
    parser.add_argument(
        "-p",
        "--path",
        default=DEFAULT_PATH,
        help=f"The path to the directory containing documents to load. Default is {DEFAULT_PATH}.",
    )
    parser.add_argument(
        "--nb-docs",
        type=int,
        default=8,
        help="The number of documents to retrieve from the vector db. Default is 8.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args.model, args.embedding_model, args.path, args.nb_docs)
