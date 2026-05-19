from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq

# env load
load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
     api_key=st.secrets["GROQ_API_KEY"]
)

# session state
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "doc_ready" not in st.session_state:
    st.session_state.doc_ready = False

# document processing
def process_document(path):

    # 1. Load PDF
    loader = PyPDFLoader(path)
    docs = loader.load()

    # 2. Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300
    )
    chunks = splitter.split_documents(docs)

    # 3. Embeddings (LOCAL - NO API COST)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Vector DB
    vector_db = FAISS.from_documents(
        chunks,
       embeddings
    )

    return vector_db


#  tool: doc search

def search_docs(query):
    return st.session_state.vector_db.similarity_search(query, k=4)


#  tool: summarize

def summarize_text(text):
    prompt = f"""Summarize the following text in a concise manner:{text}"""

    return llm.invoke(prompt).content.strip()

# tool: explain

def explain_answer(text):
    prompt = f"""Explain the following answer in a simple manner for a 5 year old:{text}"""

    return llm.invoke(prompt).content.strip()



# agent brain

def agent(query):

    # 1. search docs
    results = search_docs(query)

    # 2. create context
    context = "\n".join([d.page_content for d in results])

    # 3. create prompt
    prompt = f"""
You are a STRICT DOCUMENT QUESTION ANSWERING SYSTEM.

RULES:
1. You MUST answer ONLY using the provided context.
2. If the answer is not in the context, say:
   "I don't know based on the document."
3. Do NOT use outside knowledge.
4. Do NOT give opinions or extra information.
5. Do NOT continue conversation beyond answering.
6. Do NOT act like a chatbot or assistant.
7. Be concise and factual.

CONTEXT:
{context}

QUESTION:
{query}

FINAL ANSWER:
"""
    return llm.invoke(prompt).content.strip()



# ======================
# UI SETUP
# ======================
st.set_page_config(page_title="AI Agent RAG Bot", page_icon="🤖")

st.title("🤖 AI Agent Document System")


# ======================
# UPLOAD PAGE
# ======================
if not st.session_state.doc_ready:

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:

        file_path = "uploaded.pdf"

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("Uploaded!")

        with st.spinner("Processing document..."):
            st.session_state.vector_db = process_document(file_path)
            st.session_state.doc_ready = True

        st.success("Ready!")
        st.rerun()


# ======================
# CHAT PAGE
# ======================
else:

    st.success("📚 Document Ready")

    if st.button("Upload New Document"):
        st.session_state.doc_ready = False
        st.session_state.vector_db = None
        st.session_state.messages = []
        st.rerun()

    # CHAT HISTORY
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    query = st.chat_input("Ask anything from document...")

    if query:

        # user message
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.write(query)

        # AI response
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):
                response = agent(query)

            st.write(response)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )