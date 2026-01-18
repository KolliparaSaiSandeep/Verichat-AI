import os
import asyncio
import pathlib
import base64
import io
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from streamlit_mic_recorder import mic_recorder
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from flashrank import Ranker 
from groq import Groq
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever

load_dotenv()

# --- 1. DATA CONTRACT ---
class UnifiedAudit(BaseModel):
    is_verified: bool = Field(description="True if the answer is grounded in the files/images.")
    audit_note: str = Field(description="Summary of the audit, noting any contradictions.")

# --- 2. THE VISION ENGINE ---
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def describe_image(image_bytes):
    client = Groq()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in extreme detail for a search index. Include text, charts, and colors."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }],
        model="llama-3.2-11b-vision-preview",
    )
    return chat_completion.choices[0].message.content

# --- 3. THE UNIVERSAL ENGINE ---
class VeriChatUniversal:
    def __init__(self, file_map):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.retrievers = {}
        
        cache_dir = pathlib.Path.home() / ".flashrank_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        ranker_client = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=str(cache_dir))
        compressor = FlashrankRerank(client=ranker_client, top_n=3)

        for label, data in file_map.items():
            if label.endswith('.pdf'):
                loader = PyPDFLoader(data)
                docs = loader.load()
            else: 
                from langchain_core.documents import Document
                docs = [Document(page_content=data, metadata={"source": label})]
            
            splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
            store = Chroma.from_documents(splits, self.embeddings, collection_name=f"col_{hash(label)}")
            self.retrievers[label] = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=store.as_retriever(search_kwargs={"k": 3})
            )

        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        self.auditor = self.llm.with_structured_output(UnifiedAudit)

    async def run_universal_query(self, query):
        combined_context = ""
        for label, retriever in self.retrievers.items():
            docs = retriever.invoke(query)
            text = "\n".join([f"[Source: {label}] {d.page_content}" for d in docs])
            combined_context += f"\n{text}\n"

        prompt = f"Context:\n{combined_context}\n\nQuestion: {query}\nAnswer:"
        full_ans = ""
        placeholder = st.empty()
        async for chunk in self.llm.astream(prompt):
            full_ans += chunk.content
            placeholder.markdown(full_ans + "▌")
        placeholder.markdown(full_ans)
        
        audit = await self.auditor.ainvoke(f"Context: {combined_context}\nAnswer: {full_ans}")
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"\n[{timestamp}] QUERY: {query}\nAUDIT: {'✅' if audit.is_verified else '❌'} {audit.audit_note}\nANSWER: {full_ans}\n{'-'*50}"
        
        return full_ans, audit.audit_note, audit.is_verified, log_entry

# --- 4. THE TRANSCRIPTION LOGIC ---
def transcribe_audio(audio_bytes):
    try:
        client = Groq()
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "recording.wav"
        return client.audio.transcriptions.create(file=audio_file, model="whisper-large-v3", response_format="text")
    except Exception as e:
        st.error(f"Mic Transcription Failed: {e}")
        return None

# --- 5. UI LAYER ---
st.set_page_config(page_title="VeriChat Universal", layout="wide", page_icon="👁️")

st.markdown("""<style>
    .audit-card { padding: 15px; border-radius: 10px; background-color: #0b0d14; margin-top: 10px; }
</style>""", unsafe_allow_html=True)

st.title("👁️ VeriChat Universal Agent")

# Initialize Session States
if "msgs" not in st.session_state: st.session_state.msgs = []
if "audit_trail" not in st.session_state: 
    st.session_state.audit_trail = "VERICHAT SESSION AUDIT LOG\n" + ("="*30) + "\n"

with st.sidebar:
    st.header("Multi-Modal Vault")
    uploaded_files = st.file_uploader("Upload PDFs or Images", type=["pdf", "jpg", "png", "jpeg"], accept_multiple_files=True)
    
    if uploaded_files:
        if "engine" not in st.session_state or len(uploaded_files) != st.session_state.get("fcount", 0):
            with st.status("🏗️ Ingesting Multi-Modal Data..."):
                file_map = {}
                for f in uploaded_files:
                    if f.type == "application/pdf":
                        path = f"tmp_{f.name}"
                        with open(path, "wb") as tmp: tmp.write(f.getbuffer())
                        file_map[f.name] = path
                    else:
                        description = describe_image(f.getvalue())
                        file_map[f"Image: {f.name}"] = description
                
                st.session_state.engine = VeriChatUniversal(file_map)
                st.session_state.fcount = len(uploaded_files)
            st.success("Vault Updated")
    
    st.divider()
    st.subheader("Session Management")
    
    # CLEAR CHAT BUTTON
    if st.button("🗑️ Clear Chat & Logs", use_container_width=True):
        st.session_state.msgs = []
        st.session_state.audit_trail = "VERICHAT SESSION AUDIT LOG\n" + ("="*30) + "\n"
        st.rerun()

    # DOWNLOAD BUTTON
    st.download_button(
        label="📥 Download Audit Trail",
        data=st.session_state.audit_trail,
        file_name=f"audit_log_{datetime.now().strftime('%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True
    )

# Display Chat History
for m in st.session_state.msgs:
    with st.chat_message(m["role"]): st.markdown(m["content"])

# User Interaction
if "engine" in st.session_state:
    c1, c2 = st.columns([1, 10])
    with c1: 
        voice = mic_recorder(start_prompt="🎤", stop_prompt="🛑", key='v')
    
    user_q = st.chat_input("Ask anything...")

    if voice and 'bytes' in voice:
        with st.spinner("Transcribing..."): 
            text = transcribe_audio(voice['bytes'])
            if text: user_q = text

    if user_q:
        st.session_state.msgs.append({"role": "user", "content": user_q})
        with st.chat_message("user"): st.markdown(user_q)
        
        with st.chat_message("assistant"):
            with st.status("🔍 Auditing Response...", expanded=False):
                ans, note, verified, log_text = asyncio.run(st.session_state.engine.run_universal_query(user_q))
            
            st.session_state.msgs.append({"role": "assistant", "content": ans})
            st.session_state.audit_trail += log_text
            
            border_color = "#00ffcc" if verified else "#ff4b4b"
            st.markdown(f'''
                <div class="audit-card" style="border-left: 5px solid {border_color};">
                    <b>🛡️ Audit Report:</b><br>{note}
                </div>
            ''', unsafe_allow_html=True)
else:
    st.info("💡 Please upload a document or image in the sidebar to begin.")