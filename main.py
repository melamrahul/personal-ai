from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, hashlib
from dotenv import load_dotenv
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec

# -------------------------------
# 1️⃣ Load environment variables
# -------------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# -------------------------------
# 2️⃣ Initialize Pinecone (lightweight version)
# -------------------------------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "rag-learning-lite"

if index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=768,  # Gemini embedding dimension
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
    allow_origins=["*"],  # You can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 4️⃣ Data model
# -------------------------------
class Message(BaseModel):
    text: str

# -------------------------------
# 5️⃣ Helper – get embedding from Gemini
# -------------------------------
def get_embedding(text: str):
    result = genai.embed_content(
        model="models/embedding-001",
        content=text,
    )
    return result["embedding"]

# -------------------------------
# 6️⃣ Learn endpoint
# -------------------------------
@app.post("/learn")
async def learn(data: Message):
    """Teach the AI from text directly via prompt."""
    text = data.text.strip()
    if not text:
        return {"error": "No text provided"}

    # Split into chunks
    chunks = [text[i:i + 500] for i in range(0, len(text), 500)]
    vectors = []

    for chunk in chunks:
        chunk_id = hashlib.md5(chunk.encode()).hexdigest()  # unique ID for smart updating
        embedding = get_embedding(chunk)
        vectors.append({
            "id": chunk_id,
            "values": embedding,
            "metadata": {"text": chunk}
        })

    index.upsert(vectors=vectors)
    return {"status": "learned_or_updated", "chunks": len(chunks)}

# -------------------------------
# 7️⃣ Ask endpoint
# -------------------------------
@app.post("/ask")
async def ask(data: Message):
    """Ask a question based on learned memory."""
    question = data.text.strip()
    if not question:
        return {"error": "No question provided"}

    q_embed = get_embedding(question)
    results = index.query(vector=q_embed, top_k=3, include_metadata=True)

    if not results.matches:
        return {"answer": "I don’t know yet. Try teaching me first with 'learn'."}

    context = "\n".join([m.metadata["text"] for m in results.matches])

    prompt = f"""Use the following context to answer:
    {context}

    Question: {question}
    """

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)

    return {"answer": response.text}

# -------------------------------
# ✅ Run locally:
# python -m uvicorn main:app --host 0.0.0.0 --port 8000
# -------------------------------
