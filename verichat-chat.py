import os
import asyncio
import pathlib
import streamlit as st
from datetime import datetime # Added for timestamps
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Core LangChain & Community Imports
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from flashrank import Ranker 

load_dotenv()

# --- 1. DATA CONTRACT ---
class AuditResult(BaseModel):
    is_supported: bool = Field(description="True if the answer is grounded in the PDF context.")
    reasoning: str = Field(description="Brief explanation of the fact-check result.")

# --- 2. THE ENTERPRISE ENGINE ---
class VeriChatAdvanced:
    def __init__(self, pdf_path):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)
        self.vectorstore = Chroma.from_documents(splits, self.embeddings)
        
        # WINDOWS PERMISSION FIX
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
        context_blocks = [f"[Page {d.metadata.get('page', '?')}] {d.page_content}" for d in docs]
        context = "\n\n".join(context_blocks)
        
        attempts = 0
        feedback = ""
        full_audit_history = [] # This will store the audit trail
        
        while attempts < 2:
            prompt = f"Context: {context}\nQuestion: {query}\n{feedback}\nAnswer:"
            full_content = ""
            placeholder = st.empty()
            
            async for chunk in self.llm.astream(prompt):
                full_content += chunk.content
                placeholder.markdown(full_content + "▌")
            placeholder.markdown(full_content)
            
            # Step: Perform Audit
            audit = await self.auditor.ainvoke(f"Context: {context}\nAnswer: {full_content}")
            
            # Record the attempt in history
            log_entry = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "attempt": attempts + 1,
                "is_verified": audit.is_supported,
                "reasoning": audit.reasoning,
                "response_draft": full_content
            }
            full_audit_history.append(log_entry)
            
            if audit.is_supported:
                return full_content, audit.reasoning, full_audit_history
            
            feedback = f"\nERROR DETECTED: {audit.reasoning}. REWRITE FOR ACCURACY."
            attempts += 1
            st.warning(f"🔄 Self-Correction Triggered (Attempt {attempts})")
            
        return "Unverified answer.", "Audit failed.", full_audit_history

# --- 3. UI LAYER ---
st.set_page_config(page_title="VeriChat Enterprise", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    .audit-box { padding: 15px; border-radius: 10px; background-color: #0b0d14; border-left: 5px solid #00ffcc; margin-top: 10px; }
    .log-item { font-size: 0.85rem; border-bottom: 1px solid #3e4259; padding: 5px 0; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ VeriChat Enterprise")
st.caption("Secured Agentic RAG | Audit Trails Enabled")

with st.sidebar:
    st.header("Terminal Controls")
    uploaded_file = st.file_uploader("Upload Knowledge PDF", type="pdf")
    
    if uploaded_file:
        if "engine" not in st.session_state or st.session_state.get("fid") != uploaded_file.name:
            with st.status("🏗️ Ingesting Document...", expanded=True):
                temp_path = "active_vault.pdf"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.engine = VeriChatAdvanced(temp_path)
                st.session_state.fid = uploaded_file.name
                st.session_state.global_logs = "" # Clear logs for new file
        
        st.success(f"System Ready")
        
        # Download Button in Sidebar
        if "global_logs" in st.session_state and st.session_state.global_logs:
            st.divider()
            st.download_button(
                label="📥 Download Audit Trail (.txt)",
                data=st.session_state.global_logs,
                file_name=f"audit_trail_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain"
            )

if "messages" not in st.session_state: st.session_state.messages = []
if "global_logs" not in st.session_state: st.session_state.global_logs = "VERICHAT AUDIT LOG\n" + ("="*20) + "\n"

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if "engine" in st.session_state:
    if user_query := st.chat_input("Query the vault..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"): st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.status("🔍 Auditing Response...", expanded=False):
                ans, reason, logs = asyncio.run(st.session_state.engine.run_pipeline(user_query))
            
            # Update Global Logs for Download
            log_text = f"\n\nQUERY: {user_query}\n"
            for entry in logs:
                status_icon = "✅" if entry['is_verified'] else "❌"
                log_text += f"[{entry['timestamp']}] Attempt {entry['attempt']}: {status_icon}\n"
                log_text += f"Reason: {entry['reasoning']}\n"
            st.session_state.global_logs += log_text
            
            st.session_state.messages.append({"role": "assistant", "content": ans})
            st.markdown(f'<div class="audit-box"><b>🛡️ Auditor Log:</b><br>{reason}</div>', unsafe_allow_html=True)
else:
    st.info("Please upload a PDF to unlock the interface.")