from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import os, hashlib, traceback, uuid
from dotenv import load_dotenv
import cohere
from pinecone import Pinecone, ServerlessSpec

# -------------------------------
# 1️⃣ Load environment variables
# -------------------------------
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not COHERE_API_KEY or not PINECONE_API_KEY:
    raise ValueError("❌ Missing API keys! Check .env file for COHERE_API_KEY and PINECONE_API_KEY.")

co = cohere.Client(COHERE_API_KEY)

# -------------------------------
# 2️⃣ Initialize Pinecone
# -------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "rag-learning-lite"
expected_dim = 1024  # Cohere embed-english-v3.0 dimension

# Check if index exists and fix dimension mismatch
existing_indexes = {i.name: i for i in pc.list_indexes()}

if index_name in existing_indexes:
    info = pc.describe_index(index_name)
    if info.dimension != expected_dim:
        print(f"⚠️ Index dimension mismatch ({info.dimension} vs {expected_dim}) — recreating.")
        pc.delete_index(index_name)
        pc.create_index(
            name=index_name,
            dimension=expected_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("✅ Index recreated.")
else:
    print(f"ℹ️ Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=expected_dim,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("✅ Index created.")

index = pc.Index(index_name)

# -------------------------------
# 3️⃣ FastAPI setup
# -------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # loosen for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 4️⃣ Error logging middleware
# -------------------------------
@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        print("\n⚠️ ERROR in request:", request.url.path)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

# -------------------------------
# 5️⃣ Serve HTML (optional)
# -------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h2>✅ Backend running — try POST /learn or /ask</h2>"

# -------------------------------
# 6️⃣ Data model
# -------------------------------
class Message(BaseModel):
    text: str

# -------------------------------
# 7️⃣ Helper – get embedding
# -------------------------------
def get_embedding(text: str, input_type: str = "search_document"):
    response = co.embed(
        texts=[text],
        model="embed-english-v3.0",
        input_type=input_type,  # REQUIRED by Cohere v3
    )
    return response.embeddings[0]

# -------------------------------
# 8️⃣ Learn endpoint with deduplication
# -------------------------------
@app.post("/learn")
async def learn(data: Message):
    text = data.text.strip()
    if not text:
        return {"error": "No text provided"}

    # Split text into 500-char chunks
    chunks = [text[i:i + 500] for i in range(0, len(text), 500)]
    vectors = []
    updated_count = 0
    stored_count = 0

    for chunk in chunks:
        # Get embedding for similarity check
        embedding = get_embedding(chunk, input_type="search_document")
        
        # Check for similar existing content (similarity threshold: 0.80 = 80% similar)
        duplicate_found = False
        try:
            similar_results = index.query(
                vector=embedding,
                top_k=3,
                include_metadata=True
            )
            
            matches = getattr(similar_results, "matches", None) or similar_results.get("matches", [])
            
            # Check all top matches for duplicates
            if matches and len(matches) > 0:
                for match in matches:
                    similarity_score = match.get("score", 0) if isinstance(match, dict) else getattr(match, "score", 0)
                    
                    # For Pinecone cosine similarity: higher score = more similar
                    # Score range is typically -1 to 1, where 1 is identical
                    if similarity_score > 0.80:
                        # Delete the old duplicate
                        old_id = match.get("id") if isinstance(match, dict) else getattr(match, "id")
                        old_text = match.get("metadata", {}).get("text", "") if isinstance(match, dict) else getattr(match.metadata, "text", "")
                        
                        index.delete(ids=[old_id])
                        updated_count += 1
                        duplicate_found = True
                        print(f"🔄 Deleted duplicate (similarity: {similarity_score:.3f})")
                        print(f"   Old: {old_text[:50]}...")
                        print(f"   New: {chunk[:50]}...")
        
        except Exception as e:
            print(f"⚠️ Similarity check failed: {e}")
            # Continue with storage even if check fails
        
        # Generate unique ID and store new content
        chunk_id = str(uuid.uuid4())  # Use UUID instead of MD5 hash
        
        vectors.append({
            "id": chunk_id,
            "values": embedding,
            "metadata": {"text": chunk}
        })
        stored_count += 1

    # Upsert all vectors at once
    if vectors:
        index.upsert(vectors=vectors)

    response = {
        "status": "success",
        "chunks_stored": stored_count,
    }
    
    if updated_count > 0:
        response["chunks_updated"] = updated_count
        response["message"] = f"Updated {updated_count} similar entries"
    
    return response

# -------------------------------
# 9️⃣ Ask endpoint
# -------------------------------
@app.post("/ask")
async def ask(data: Message):
    question = data.text.strip()
    if not question:
        return {"error": "No question provided"}

    q_embed = get_embedding(question, input_type="search_query")
    results = index.query(vector=q_embed, top_k=3, include_metadata=True)

    matches = getattr(results, "matches", None) or results.get("matches", [])

    if not matches:
        return {"answer": "I don't know yet. Try teaching me first with /learn."}

    # Extract text context
    context = "\n".join([
        m["metadata"]["text"] if isinstance(m, dict) else m.metadata["text"]
        for m in matches
    ])

    prompt = f"""
    Use the following context to answer the question.

    Context:
    {context}

    Question:
    {question}
    """

    # Updated Cohere chat API
    chat_response = co.chat(model="command-r-plus-08-2024", message=prompt)

    # Defensive response extraction
    answer_text = getattr(chat_response, "text", None)
    if not answer_text:
        answer_text = str(chat_response)

    return {"answer": answer_text.strip()}

# -------------------------------
# ✅ Run locally
# python -m uvicorn main:app --host 0.0.0.0 --port 8000
# -------------------------------
