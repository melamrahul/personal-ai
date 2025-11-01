from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec

# -------------------------------
# 1️⃣ Load environment variables
# -------------------------------
load_dotenv()

# -------------------------------
# 2️⃣ Configure Google Gemini API
# -------------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# -------------------------------
# 3️⃣ Initialize Pinecone
# -------------------------------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "rag-learning-demo"

# Create index if it doesn’t exist
existing_indexes = [i.name for i in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to the index
index = pc.Index(index_name)

# -------------------------------
# 4️⃣ Initialize Embedder
# -------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# 5️⃣ Initialize FastAPI
# -------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can later restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 6️⃣ Data Models
# -------------------------------
class Query(BaseModel):
    question: str


# -------------------------------
# 7️⃣ Learning Endpoint
# -------------------------------
@app.post("/train")
async def train(data: dict):
    """Learn directly from input text"""
    text = data.get("text", "")
    if not text.strip():
        return {"error": "No text provided"}

    chunks = [text[i:i + 500] for i in range(0, len(text), 500)]
    embeddings = embedder.encode(chunks).tolist()

    vectors = [
        {"id": f"chunk-{i}", "values": embeddings[i], "metadata": {"text": chunks[i]}}
        for i in range(len(chunks))
    ]

    index.upsert(vectors=vectors)
    return {"status": "trained", "chunks_stored": len(chunks)}


# -------------------------------
# 8️⃣ Question Answering Endpoint
# -------------------------------
@app.post("/ask")
async def ask(query: Query):
    """Retrieve and answer questions based on learned data"""
    q_embed = embedder.encode(query.question).tolist()
    results = index.query(vector=q_embed, top_k=3, include_metadata=True)
    context = "\n".join([m["metadata"]["text"] for m in results["matches"]])

    prompt = f"""
    Answer the question using the context below:
    Context:
    {context}

    Question:
    {query.question}
    """

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return {"answer": response.text}


# -------------------------------
# ✅ 9️⃣ Run for Render
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Required for Render
    uvicorn.run("main:app", host="0.0.0.0", port=port)
