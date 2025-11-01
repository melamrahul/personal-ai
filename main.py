from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
        dimension=1024,  # Cohere embedding dimension
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
# 5️⃣ Helper – get embedding from Cohere
# -------------------------------
def get_embedding(text: str):
    response = co.embed(texts=[text], model="embed-english-v3.0")
    return response.embeddings[0]

# -------------------------------
# 6️⃣ Learn endpoint
# -------------------------------
@app.post("/learn")
async def learn(data: Message):
    text = data.text.strip()
    if not text:
        return {"error": "No text provided"}

    # Split into chunks
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
# 7️⃣ Ask endpoint
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

    # Use Cohere chat model for answer synthesis
    prompt = f"""
    Use the following context to answer the question:

    Context:
    {context}

    Question:
    {question}
    """

    response = co.chat(
        model="command-r-plus",
        message=prompt
    )

    return {"answer": response.text.strip()}

# -------------------------------
# ✅ Run locally:
# python -m uvicorn main:app --host 0.0.0.0 --port 8000
# -------------------------------
