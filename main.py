from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os, hashlib
from dotenv import load_dotenv
import cohere
from pinecone import Pinecone, ServerlessSpec

# -------------------------------
# 1️⃣ Load environment variables
# -------------------------------
load_dotenv()
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# -------------------------------
# 2️⃣ Initialize Pinecone
# -------------------------------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "rag-learning-lite"

if index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1024,  # Cohere embed-english-v3.0 dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# -------------------------------
# 3️⃣ FastAPI App setup
# -------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # loosen for testing, tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 4️⃣ Serve your HTML (optional)
# -------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h2>Backend running — try POST /learn or /ask</h2>"

# -------------------------------
# 5️⃣ Data model
# -------------------------------
class Message(BaseModel):
    text: str

# -------------------------------
# 6️⃣ Helper – get embedding
# -------------------------------
def get_embedding(text: str):
    response = co.embed(texts=[text], model="embed-english-v3.0")
    return response.embeddings[0]

# -------------------------------
# 7️⃣ Learn endpoint
# -------------------------------
@app.post("/learn")
async def learn(data: Message):
    text = data.text.strip()
    if not text:
        return {"error": "No text provided"}

    # Split text into 500-char chunks
    chunks = [text[i:i + 500] for i in range(0, len(text), 500)]
    vectors = []

    for chunk in chunks:
        chunk_id = hashlib.md5(chunk.encode()).hexdigest()
        embedding = get_embedding(chunk)
        vectors.append({
            "id": chunk_id,
            "values": embedding,
            "metadata": {"text": chunk}
        })

    index.upsert(vectors=vectors)
    return {"status": "learned_or_updated", "chunks_stored": len(chunks)}

# -------------------------------
# 8️⃣ Ask endpoint
# -------------------------------
@app.post("/ask")
async def ask(data: Message):
    question = data.text.strip()
    if not question:
        return {"error": "No question provided"}

    q_embed = get_embedding(question)
    results = index.query(vector=q_embed, top_k=3, include_metadata=True)

    if not results.matches:
        return {"answer": "I don’t know yet. Try teaching me first with /learn."}

    context = "\n".join([m.metadata["text"] for m in results.matches])

    prompt = f"""
    Use the following context to answer the question:

    Context:
    {context}

    Question:
    {question}
    """

    # Use Cohere chat model correctly
    response = co.chat(
        model="command-r-plus",
        message=prompt  # ✅ 'message' is correct param name
    )

    return {"answer": response.text.strip()}

# -------------------------------
# ✅ Run locally:
# python -m uvicorn main:app --host 0.0.0.0 --port 8000
# -------------------------------
