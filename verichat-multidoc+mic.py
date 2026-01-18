import os
import asyncio
import pathlib
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from streamlit_mic_recorder import mic_recorder

# Core LangChain & Community Imports
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from flashrank import Ranker 
from groq import Groq

load_dotenv()

# --- 1. DATA CONTRACT ---
class UnifiedAudit(BaseModel):
    is_verified: bool = Field(description="True if the answer is grounded in the provided files.")
    audit_note: str = Field(description="Summary of the audit, noting any contradictions found.")

# --- 2. THE UNIVERSAL ENGINE ---
class VeriChatUniversal:
    def __init__(self, pdf_paths):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.retrievers = {}
        
        # Windows Permission Fix
        cache_dir = pathlib.Path.home() / ".flashrank_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        ranker_client = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=str(cache_dir))
        compressor = FlashrankRerank(client=ranker_client, top_n=3)

        for label, path in pdf_paths.items():
            loader = PyPDFLoader(path)
            docs = loader.load()
            splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
            
            # Unique collection for every file
            store = Chroma.from_documents(splits, self.embeddings, collection_name=f"col_{hash(label)}")
            base = store.as_retriever(search_kwargs={"k": 3})
            self.retrievers[label] = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=base
            )

        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        self.auditor = self.llm.with_structured_output(UnifiedAudit)

    async def run_universal_query(self, query):
        combined_context = ""
        for label, retriever in self.retrievers.items():
            docs = retriever.invoke(query)
            text = "\n".join([f"[Source: {label}, Page {d.metadata.get('page','?')}] {d.page_content}" for d in docs])
            combined_context += f"\n{text}\n"

        prompt = f"Using the context below, answer the question. If multiple files are present, compare them.\n\nContext:\n{combined_context}\n\nQuestion: {query}\nAnswer:"
        
        full_ans = ""
        placeholder = st.empty()
        async for chunk in self.llm.astream(prompt):
            full_ans += chunk.content
            placeholder.markdown(full_ans + "▌")
        placeholder.markdown(full_ans)
        
        audit = await self.auditor.ainvoke(f"Context: {combined_context}\nAnswer: {full_ans}")
        return full_ans, audit.audit_note, audit.is_verified

# --- 3. THE TRANSCRIPTION LOGIC ---
def transcribe_audio(audio_bytes):
    client = Groq()
    with open("tmp_audio.wav", "wb") as f: f.write(audio_bytes)
    with open("tmp_audio.wav", "rb") as file:
        return client.audio.transcriptions.create(file=("audio.wav", file.read()), model="whisper-large-v3", response_format="text")

# --- 4. THE UI LAYER ---
st.set_page_config(page_title="VeriChat Universal", layout="wide", page_icon="🌐")
st.markdown("""<style>.audit-card { padding: 15px; border-radius: 10px; background-color: #0b0d14; border-left: 5px solid #00ffcc; }</style>""", unsafe_allow_html=True)

st.title("🌐 VeriChat Universal Agent")
st.caption("Agentic RAG | Single/Multi-Doc Support | Voice Enabled")

with st.sidebar:
    st.header("Document Center")
    files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    if files:
        if "engine" not in st.session_state or len(files) != st.session_state.get("fcount", 0):
            with st.status("🏗️ Rebuilding Knowledge Vault..."):
                file_map = {f.name: f"tmp_{f.name}" for f in files}
                for f in files:
                    with open(file_map[f.name], "wb") as tmp: tmp.write(f.getbuffer())
                st.session_state.engine = VeriChatUniversal(file_map)
                st.session_state.fcount = len(files)
        st.success(f"{len(files)} Document(s) Online")

if "msgs" not in st.session_state: st.session_state.msgs = []

for m in st.session_state.msgs:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if "engine" in st.session_state:
    c1, c2 = st.columns([1, 10])
    with c1: voice = mic_recorder(start_prompt="🎤", stop_prompt="🛑", key='v')
    user_q = st.chat_input("Ask anything...")

    if voice and 'bytes' in voice:
        with st.spinner("Transcribing..."): user_q = transcribe_audio(voice['bytes'])

    if user_q:
        st.session_state.msgs.append({"role": "user", "content": user_q})
        with st.chat_message("user"): st.markdown(user_q)
        with st.chat_message("assistant"):
            with st.status("🔍 Auditing Response...", expanded=False):
                ans, note, verified = asyncio.run(st.session_state.engine.run_universal_query(user_q))
            st.session_state.msgs.append({"role": "assistant", "content": ans})
            st.markdown(f'<div class="audit-card"><b>🛡️ Audit Report:</b><br>{note}</div>', unsafe_allow_html=True)