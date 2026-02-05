import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ----------------------------
# Setup
# ----------------------------
load_dotenv()

st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="📺",
    layout="centered"
)

st.title("📺 YouTube RAG Chatbot")
st.caption("Ask questions based only on the video transcript")

# ----------------------------
# Load LLM
# ----------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2
)

# ----------------------------
# Load Vector Store
# ----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "data/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ----------------------------
# Prompt
# ----------------------------
prompt = PromptTemplate(
    template="""
You are a helpful assistant.

Using ONLY the transcript context below:
- First answer YES or NO clearly
- Then summarize what was discussed in bullet points
- Do NOT add information not present in the transcript

Transcript Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

# ----------------------------
# User Input
# ----------------------------
question = st.text_input(
    "Ask a question about the video:",
    placeholder="Was the topic of LLM discussed in the video?"
)

# ----------------------------
# Generate Answer
# ----------------------------
if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join(doc.page_content for doc in docs)

        final_prompt = prompt.format(
            context=context,
            question=question
        )

        response = llm.invoke(final_prompt)

    st.markdown("### 🧠 Answer")
    st.write(response.content)
