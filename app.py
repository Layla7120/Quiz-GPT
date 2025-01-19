import json
from pathlib import Path

import streamlit as st
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser
from langchain.text_splitter import CharacterTextSplitter

from prompts import formatting_prompt, questions_prompt

Path("./.cache/quiz_files").mkdir(parents=True, exist_ok=True)

class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()

st.set_page_config(
    page_title="Quiz GPT",
    page_icon="🤔"
)

@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


with st.sidebar:
    st.title("OpenAI API KEY")
    API_KEY = st.text_input("Use your API KEY", type="password")

    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        )
    )

    docs = None
    if choice == "File":
        file = st.file_uploader(
            "Upload a .txt, .pdf or .docx file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
            st.write(docs)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            retriever = WikipediaRetriever(top_k_results=3)
            with st.status("Searching..."):
                docs = retriever.get_relevant_documents(topic)

    st.markdown("""
    ### 🔗 Github Repo 

    [![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/Layla7120/Quiz-GPT)
    """)
    st.subheader("2025-01-19")

if API_KEY:
    try:
        llm = ChatOpenAI(
            temperature=0.1,
            model="gpt-3.5-turbo-1106",
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
            api_key=API_KEY
        )
        st.success("ChatOpenAI initialized successfully!")
    except Exception as e:
        st.error(f"Failed to initialize ChatOpenAI: {e}")
else:
    st.warning("Please enter your OpenAI API key in the sidebar to proceed.")


st.header("🤔 Quiz GPT")

if not docs or not API_KEY:
    st.markdown(
        """
        Make a quiz and check your answer!

        You can either search for information on Wikipedia or upload your own file as a resource.
        """
    )

else:
    start = st.button("Generate Quiz")

    if start:
        formatting_chain = formatting_prompt | llm
        questions_chain = {"context": format_docs} | questions_prompt | llm

        if start:
            chain = {"context": questions_chain} | formatting_chain | output_parser
            response = chain.invoke(docs)
            st.write(response)


