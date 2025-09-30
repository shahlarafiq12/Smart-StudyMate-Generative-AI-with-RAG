import streamlit as st
import sqlite3
import hashlib
import os
from dotenv import load_dotenv
from streamlit_option_menu import option_menu
from pypdf import PdfReader   # install with: pip install pypdf

# LangChain + Google GenAI
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ========================
# CONFIG
# ========================
DB_NAME = "studymate.db"
UPLOAD_DIR = "user_uploaded_files"
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

study_model = genai.GenerativeModel(
    "gemini-2.0-flash",
    system_instruction="""
    You are StudyMate, a friendly and intelligent study assistant.
    Roles:
    1. Summarize study material into simple explanations.
    2. Generate practice questions, flashcards, and MCQs.
    3. Help clarify concepts with examples.
    4. Encourage active learning, but stay focused on education.
    """
)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# ========================
# DB FUNCTIONS
# ========================
def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users(
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS files(
            file_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
        """)
    print("Database ready")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def sign_up(first_name, last_name, email, password):
    with sqlite3.connect(DB_NAME) as conn:
        try:
            conn.execute("""
            INSERT INTO users (first_name, last_name, email, password)
            VALUES (?, ?, ?, ?)
            """, (first_name, last_name, email, hash_password(password)))
            conn.commit()
            return True, "Account created successfully!"
        except sqlite3.IntegrityError:
            return False, "This email is already registered."

def login(email, password):
    with sqlite3.connect(DB_NAME) as conn:
        return conn.execute("""
        SELECT user_id, first_name, last_name FROM users 
        WHERE email = ? AND password = ?
        """, (email, hash_password(password))).fetchone()

def save_file(user_id, file_name, file_path):
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("INSERT INTO files (user_id, file_name, file_path) VALUES (?, ?, ?)",
                     (user_id, file_name, file_path))
        conn.commit()

def get_user_files(user_id):
    with sqlite3.connect(DB_NAME) as conn:
        return conn.execute("SELECT file_name, file_path FROM files WHERE user_id = ?", (user_id,)).fetchall()

def delete_file(user_id, file_name):
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("DELETE FROM files WHERE (user_id, file_name) = (?, ?)", (user_id, file_name))
        conn.commit()

init_db()

# ========================
# RAG FUNCTIONS
# ========================
def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return splitter.split_text(text)

def get_vector_store(chunks):
    return FAISS.from_texts(chunks, embeddings)

def get_rel_text(query, db):
    results = db.similarity_search(query, k=3)
    return [r.page_content for r in results]

# ========================
# STREAMLIT UI
# ========================
st.set_page_config(page_title="StudyMate", page_icon="📚", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = {}

with st.sidebar:
    selected = option_menu(
        "Menu",
        ["Landing Page", "Login/Signup", "Study Assistant Bot", "Notes Bot"],
        icons=["house", "person", "chat-dots", "book"],
        menu_icon="cast",
        default_index=0
    )

# LANDING PAGE
if selected == "Landing Page":
    st.title("📚 Welcome to Smart StudyMate")
    st.write("""
    **Smart StudyMate** is your personal AI-powered study assistant.
    This system is designed to make learning easier, smarter, and more interactive.
    
    ### 🔹 What you can do with StudyMate?
    - **👤 Login/Signup** – Create your account to save notes and access personalized study sessions.
    - **🤖 Study Assistant Bot** – Chat with an AI tutor that explains concepts, answers questions, and generates examples.
    - **📑 Notes Bot** – Upload your lecture notes or textbooks (PDFs), and ask questions directly from your own material.
    - **🔍 RAG (Retrieval-Augmented Generation)** – Retrieves the most relevant parts of your uploaded notes so answers are accurate and grounded.
    - **📂 File Management** – Upload, view, and delete your study files easily.
    - **📝 Practice Questions** – Get MCQs, flashcards, and quizzes generated from your notes.

    ### 🎯 Why use StudyMate?
    - Saves time by summarizing large study materials.
    - Makes learning interactive and fun.
    - Helps in exam preparation with AI-generated questions.
    - Keeps all your notes organized in one place.

    ---
    🚀 Get started by creating your account in the **Login/Signup** section!
    """)

# LOGIN / SIGNUP
if selected == "Login/Signup":
    st.header("Login / Signup")

    if "user_id" in st.session_state:
        st.info(f"Logged in as {st.session_state['first_name']} {st.session_state['last_name']}")
        if st.button("Logout"):
            st.session_state.clear()
            st.success("Logged out!")
    else:
        action = st.radio("Choose Action", ["Login", "Sign Up"])
        if action == "Sign Up":
            first = st.text_input("First Name")
            last = st.text_input("Last Name")
            email = st.text_input("Email")
            pw = st.text_input("Password", type="password")
            if st.button("Create Account"):
                ok, msg = sign_up(first, last, email, pw)
                st.success(msg) if ok else st.error(msg)

        elif action == "Login":
            email = st.text_input("Email")
            pw = st.text_input("Password", type="password")
            if st.button("Login"):
                user = login(email, pw)
                if user:
                    st.session_state['user_id'], st.session_state['first_name'], st.session_state['last_name'] = user
                    st.success(f"Welcome {user[1]}!")
                    st.session_state.messages[user[0]] = []
                else:
                    st.error("Invalid credentials.")

# STUDY ASSISTANT BOT
if selected == "Study Assistant Bot":
    st.subheader("Chat with StudyMate")
    if "user_id" not in st.session_state:
        st.warning("Please login first.")
    else:
        chat_history = st.session_state.messages.get(st.session_state['user_id'], [])
        chat_bot = study_model.start_chat(history=chat_history)

        # show history
        for msg in chat_history:
            who = "user" if msg["role"] == "user" else "assistant"
            st.chat_message(who).markdown(msg["parts"][0])

        q = st.chat_input("Ask a study question...")
        if q:
            st.chat_message("user").markdown(q)
            chat_history.append({"role": "user", "parts": [q]})

            with st.spinner("Thinking..."):
                res = chat_bot.send_message(q)
                st.chat_message("assistant").markdown(res.text)
                chat_history.append({"role": "model", "parts": [res.text]})

            st.session_state.messages[st.session_state['user_id']] = chat_history

# NOTES BOT
if selected == "Notes Bot":
    st.subheader("Upload and Query Notes")
    if "user_id" not in st.session_state:
        st.warning("Please login first.")
    else:
        option = st.radio("Choose", ["Upload Notes", "Chat with Notes"])
        if option == "Upload Notes":
            file = st.file_uploader("Upload PDF Notes", type="pdf")
            if file:
                file_name = file.name
                file_path = os.path.join(UPLOAD_DIR, f"{st.session_state['user_id']}_{file_name}")
                os.makedirs(UPLOAD_DIR, exist_ok=True)
                with open(file_path, "wb") as f:
                    f.write(file.read())
                if st.button("Save"):
                    save_file(st.session_state['user_id'], file_name, file_path)
                    st.success("File saved!")

            files = get_user_files(st.session_state['user_id'])
            if files:
                st.subheader("Your Files")
                for fname, fpath in files:
                    st.markdown(f"- {fname}")
                    if st.button(f"Delete {fname}"):
                        delete_file(st.session_state['user_id'], fname)
                        if os.path.exists(fpath): os.remove(fpath)
                        st.success("Deleted!")

        elif option == "Chat with Notes":
            files = get_user_files(st.session_state['user_id'])
            if not files:
                st.info("Upload notes first.")
            else:
                selected_file = st.selectbox("Select File", [f[0] for f in files])
                if st.button("Process File"):
                    file_path = [f[1] for f in files if f[0] == selected_file][0]
                    pdf = PdfReader(file_path)
                    text = "".join([p.extract_text() for p in pdf.pages])
                    chunks = get_chunks(text)
                    db = get_vector_store(chunks)
                    st.session_state['notes_db'] = db
                    st.success("File processed!")

                if "notes_db" in st.session_state:
                    q = st.chat_input("Ask something about your notes...")
                    if q:
                        rel_texts = get_rel_text(q, st.session_state['notes_db'])
                        prompt = f"Context: {' '.join(rel_texts)}\n\nQuestion: {q}\nAnswer as a tutor:"
                        res = study_model.generate_content(prompt)
                        st.chat_message("assistant").markdown(res.text)
