from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
#from langchain_community.retrievers import MultiQueryRetriever
from dotenv import load_dotenv
load_dotenv()

# Paths
TEXT_PATH = Path("../data/text/video.txt")

# Read text
text = TEXT_PATH.read_text(encoding="utf-8")

# Splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=60
)

chunks = splitter.split_text(text)

'''print(len(chunks))
for idx, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {idx} ---\n")
    print(chunk)'''

embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_texts(
    texts=chunks,
    embedding=embeddings
)

#print(vector_store.index_to_docstore_id)

retriever=vector_store.as_retriever(search_kwargs={"k": 4})


prompt = PromptTemplate(
    template="""
You are a helpful assistant.

Using ONLY the transcript context below:
- Answer clearly and completely
- If the topic is discussed, summarize the key points in bullet points
- Do NOT add information that is not in the transcript

Transcript Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)


question="was the topiv of llm was discussed in the video? If yes then what was discussed ?"
relevant_chunks=retriever.invoke(question)

'''for i, doc in enumerate(relevant_chunks, 1):
    print(f"\n--- Retrieved Chunk {i} ---\n")
    print(doc.page_content[:300])
'''

context = "\n\n".join(doc.page_content for doc in relevant_chunks)

final_prompt = prompt.format(
    context=context,
    question=question
)

#print(final_prompt)

llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0.2
)

answer=llm.invoke(final_prompt)
print(answer.content)