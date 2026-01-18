import os
import asyncio
import pathlib
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from streamlit_mic_recorder import mic_recorder # New Voice Component

# Core LangChain & Community Imports
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from flashrank import Ranker 
from groq import Groq # Used for Whisper Transcription

load_dotenv()

# --- 1. DATA CONTRACT & ENGINE (Keeping your existing logic) ---
class AuditResult(BaseModel):
    is_supported: bool = Field(description="True if the answer is grounded in the PDF.")
    reasoning: str = Field(description="Brief explanation of the fact-check result.")

class VeriChatAdvanced:
    def __init__(self, pdf_path):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)
        self.vectorstore = Chroma.from_documents(splits, self.embeddings)
        
        cache_dir = pathlib.Path.home() / ".flashrank_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        ranker_client = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=str(cache_dir))
        compressor = FlashrankRerank(client=ranker_client, top_n=3)
        
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=self.vectorstore.as_retriever(search_kwargs={"k": 6})
        )
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        self.auditor = self.llm.with_structured_output(AuditResult)

    async def run_pipeline(self, query):
        docs = self.retriever.invoke(query)
        context = "\n\n".join([f"[Page {d.metadata.get('page', '?')}] {d.page_content}" for d in docs])
        attempts, feedback, history = 0, "", []
        
        while attempts < 2:
            prompt = f"Context: {context}\nQuestion: {query}\n{feedback}\nAnswer:"
            full_content = ""
            placeholder = st.empty()
            async for chunk in self.llm.astream(prompt):
                full_content += chunk.content
                placeholder.markdown(full_content + "▌")
            placeholder.markdown(full_content)
            
            audit = await self.auditor.ainvoke(f"Context: {context}\nAnswer: {full_content}")
            history.append({"attempt": attempts+1, "verified": audit.is_supported, "reason": audit.reasoning})
            
            if audit.is_supported: return full_content, audit.reasoning, history
            feedback = f"\nERROR: {audit.reasoning}. REWRITE."
            attempts += 1
            st.warning("🔄 Self-Correction Active...")
        return "Unverified.", "Audit failed.", history

# --- 2. THE VOICE TRANSCRIPTION LOGIC ---
def transcribe_audio(audio_bytes):
    client = Groq()
    # Save temp audio file
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_bytes)
    
    with open("temp_audio.wav", "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=("temp_audio.wav", file.read()),
            model="whisper-large-v3",
            response_format="text",
        )
    return transcription

# --- 3. IMPROVISED UI LAYER ---
st.set_page_config(page_title="VeriChat Voice", layout="wide", page_icon="🎙️")
st.markdown("""<style>.audit-box { padding: 15px; border-radius: 10px; background-color: #0b0d14; border-left: 5px solid #00ffcc; }</style>""", unsafe_allow_html=True)

st.title("🎙️ VeriChat Voice Enterprise")
st.caption("Secured Agentic RAG | Talk to your Documents")

with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file:
        if "engine" not in st.session_state or st.session_state.get("fid") != uploaded_file.name:
            with st.status("🏗️ Ingesting..."):
                with open("vault.pdf", "wb") as f: f.write(uploaded_file.getbuffer())
                st.session_state.engine = VeriChatAdvanced("vault.pdf")
                st.session_state.fid = uploaded_file.name
        st.success("Ready")

if "messages" not in st.session_state: st.session_state.messages = []

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

# --- VOICE & TEXT INPUT AREA ---
if "engine" in st.session_state:
    # 1. Voice Recorder in a small column
    col1, col2 = st.columns([1, 9])
    with col1:
        audio = mic_recorder(start_prompt="🎤", stop_prompt="🛑", key='recorder')
    
    # 2. Text Input
    user_query = st.chat_input("Or type your question here...")

    # Process Voice if detected
    if audio and 'bytes' in audio:
        with st.spinner("Transcribing..."):
            user_query = transcribe_audio(audio['bytes'])

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"): st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.status("🔍 Auditing...", expanded=False):
                ans, reason, logs = asyncio.run(st.session_state.engine.run_pipeline(user_query))
            st.session_state.messages.append({"role": "assistant", "content": ans})
            st.markdown(f'<div class="audit-box"><b>🛡️ Auditor Log:</b><br>{reason}</div>', unsafe_allow_html=True)