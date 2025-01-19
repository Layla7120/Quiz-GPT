import streamlit as st
from langchain.chat_models import openai, ChatOpenAI

st.set_page_config(
    page_title="Quiz GPT",
    page_icon="🤔"
)



st.markdown(
    """
    # 🤔 Quiz GPT
    
    Make a quiz and check your answer!
    
    You can either search for information on Wikipedia or upload your own file as a resource.
    """
)

with st.sidebar:
    st.title("OpenAI API KEY")
    API_KEY = st.text_input("Use your API KEY")

    file = st.file_uploader(
        "Upload a .txt, .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

    st.title("🔗 Github Repo")
    st.markdown("[![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/Layla7120/Quiz-GPT)")

    st.subheader("2025-01-19")

if API_KEY:
    try:
        llm = ChatOpenAI(
            temperature=0.1,
            streaming=True,
            callbacks=[
            #     ChatCallbackHandler(),
            ],
            api_key=API_KEY
        )
        st.success("ChatOpenAI initialized successfully!")
    except Exception as e:
        st.error(f"Failed to initialize ChatOpenAI: {e}")
else:
    st.warning("Please enter your OpenAI API key in the sidebar to proceed.")
