from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import os, traceback, uuid, time
from datetime import datetime
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
    raise ValueError("❌ Missing API keys! Check .env file")

co = cohere.Client(COHERE_API_KEY)

# -------------------------------
# 2️⃣ Initialize Pinecone
# -------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "rag-learning-lite"
expected_dim = 1024

# Ensure index exists with correct dimensions
existing_indexes = {i.name: i for i in pc.list_indexes()}

if index_name in existing_indexes:
    info = pc.describe_index(index_name)
    if info.dimension != expected_dim:
        print(f"⚠️ Recreating index due to dimension mismatch")
        pc.delete_index(index_name)
        time.sleep(1)
        pc.create_index(
            name=index_name,
            dimension=expected_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("✅ Index recreated")
else:
    print(f"ℹ️ Creating new index '{index_name}'")
    pc.create_index(
        name=index_name,
        dimension=expected_dim,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("✅ Index created")

index = pc.Index(index_name)

# Wait for index to be ready
time.sleep(1)

# -------------------------------
# 3️⃣ FastAPI setup
# -------------------------------
app = FastAPI(title="RAG Learning System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 4️⃣ Error handling middleware
# -------------------------------
@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        print(f"\n⚠️ ERROR at {request.url.path}:")
        traceback.print_exc()
        return JSONResponse(
            status_code=500, 
            content={"error": str(e), "detail": traceback.format_exc()}
        )

# -------------------------------
# 5️⃣ Data models
# -------------------------------
class Message(BaseModel):
    text: str

class LearnRequest(BaseModel):
    text: str
    chunk_size: Optional[int] = 400  # Smaller chunks for better retrieval

class AskRequest(BaseModel):
    text: str
    top_k: Optional[int] = 5  # Retrieve more context

class ForgetRequest(BaseModel):
    query: Optional[str] = None  # Specific content to forget
    forget_all: Optional[bool] = False

# -------------------------------
# 6️⃣ Helper functions
# -------------------------------
def get_embedding(text: str, input_type: str = "search_document"):
    """Get embedding from Cohere with proper error handling"""
    try:
        response = co.embed(
            texts=[text],
            model="embed-english-v3.0",
            input_type=input_type,
            truncate="END"  # Handle long texts gracefully
        )
        return response.embeddings[0]
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        raise

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks for better context"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundaries
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > chunk_size * 0.5:  # Only if we're past halfway
                chunk = chunk[:break_point + 1]
                end = start + break_point + 1
        
        chunks.append(chunk.strip())
        start = end - overlap  # Create overlap for context continuity
    
    return chunks

# -------------------------------
# 7️⃣ Learn endpoint - IMPROVED
# -------------------------------
@app.post("/learn")
async def learn(data: LearnRequest):
    """
    Store knowledge in the vector database.
    Handles deduplication intelligently - only updates if VERY similar (95%+)
    """
    text = data.text.strip()
    if not text:
        return {"error": "No text provided", "success": False}
    
    if len(text) < 10:
        return {"error": "Text too short - provide at least 10 characters", "success": False}

    # Create chunks with overlap
    chunks = chunk_text(text, chunk_size=data.chunk_size)
    
    vectors = []
    new_count = 0
    updated_count = 0
    skipped_count = 0
    
    print(f"\n📚 Learning {len(chunks)} chunks...")

    for idx, chunk in enumerate(chunks):
        try:
            # Get embedding
            embedding = get_embedding(chunk, input_type="search_document")
            
            # Check for near-duplicates (95%+ similarity = essentially the same)
            should_skip = False
            try:
                similar_results = index.query(
                    vector=embedding,
                    top_k=1,
                    include_metadata=True
                )
                
                matches = getattr(similar_results, "matches", []) or similar_results.get("matches", [])
                
                if matches:
                    best_match = matches[0]
                    similarity = best_match.get("score", 0) if isinstance(best_match, dict) else getattr(best_match, "score", 0)
                    
                    # Only consider it a duplicate if 95%+ similar (near-exact match)
                    if similarity > 0.95:
                        old_id = best_match.get("id") if isinstance(best_match, dict) else getattr(best_match, "id")
                        old_text = best_match.get("metadata", {}).get("text", "") if isinstance(best_match, dict) else getattr(best_match.metadata, {}).get("text", "")
                        
                        # If the new chunk is longer/better, replace it
                        if len(chunk) > len(old_text):
                            index.delete(ids=[old_id])
                            updated_count += 1
                            print(f"  🔄 Chunk {idx+1}: Updated (similarity: {similarity:.3f})")
                        else:
                            skipped_count += 1
                            should_skip = True
                            print(f"  ⏭️  Chunk {idx+1}: Skipped (already exists, similarity: {similarity:.3f})")
            
            except Exception as e:
                print(f"  ⚠️ Similarity check failed for chunk {idx+1}: {e}")
            
            # Store if not skipped
            if not should_skip:
                chunk_id = str(uuid.uuid4())
                timestamp = datetime.now().isoformat()
                
                vectors.append({
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": {
                        "text": chunk,
                        "timestamp": timestamp,
                        "length": len(chunk)
                    }
                })
                new_count += 1
                print(f"  ✅ Chunk {idx+1}: Stored")
        
        except Exception as e:
            print(f"  ❌ Error processing chunk {idx+1}: {e}")
            continue

    # Upsert all new vectors
    if vectors:
        try:
            index.upsert(vectors=vectors)
            print(f"✅ Successfully stored {len(vectors)} vectors")
        except Exception as e:
            print(f"❌ Upsert failed: {e}")
            return {"error": f"Failed to store vectors: {str(e)}", "success": False}

    # Get stats
    try:
        stats = index.describe_index_stats()
        total_vectors = stats.total_vector_count if hasattr(stats, 'total_vector_count') else stats.get('total_vector_count', 0)
    except:
        total_vectors = "unknown"

    response = {
        "success": True,
        "chunks_processed": len(chunks),
        "chunks_stored": new_count,
        "chunks_updated": updated_count,
        "chunks_skipped": skipped_count,
        "total_knowledge_items": total_vectors,
        "message": f"Learned successfully! {new_count} new, {updated_count} updated, {skipped_count} skipped (duplicates)"
    }
    
    return response

# -------------------------------
# 8️⃣ Ask endpoint - IMPROVED
# -------------------------------
@app.post("/ask")
async def ask(data: AskRequest):
    """
    Answer questions using learned knowledge.
    Returns clear indication if answer is from learned data or not.
    """
    question = data.text.strip()
    if not question:
        return {"error": "No question provided", "success": False}

    print(f"\n❓ Question: {question}")

    try:
        # Get query embedding
        q_embed = get_embedding(question, input_type="search_query")
        
        # Search for relevant content
        results = index.query(
            vector=q_embed, 
            top_k=data.top_k, 
            include_metadata=True
        )

        matches = getattr(results, "matches", []) or results.get("matches", [])

        if not matches:
            return {
                "answer": "I don't have any learned knowledge yet. Please teach me something using the /learn endpoint first!",
                "source": "none",
                "confidence": 0,
                "success": True
            }

        # Check relevance of best match
        best_score = matches[0].get("score", 0) if isinstance(matches[0], dict) else getattr(matches[0], "score", 0)
        
        print(f"🔍 Found {len(matches)} matches, best score: {best_score:.3f}")

        # If best match is low quality, indicate uncertainty
        if best_score < 0.3:
            return {
                "answer": "I found some information but it doesn't seem very relevant to your question. Could you rephrase or teach me more about this topic?",
                "source": "low_confidence",
                "confidence": best_score,
                "success": True
            }

        # Extract context from matches
        context_parts = []
        for i, match in enumerate(matches):
            score = match.get("score", 0) if isinstance(match, dict) else getattr(match, "score", 0)
            text = match.get("metadata", {}).get("text", "") if isinstance(match, dict) else getattr(match, "metadata", {}).get("text", "")
            
            if score > 0.2:  # Only use reasonably relevant matches
                context_parts.append(f"[Context {i+1}, relevance: {score:.2f}]\n{text}")
        
        context = "\n\n".join(context_parts)

        # Create focused prompt
        prompt = f"""You are a helpful assistant answering based on learned knowledge.

Question: {question}

Relevant information from my knowledge base:
{context}

Instructions:
- Answer the question directly and clearly using the context provided
- If the context fully answers the question, be confident
- If the context only partially answers it, acknowledge what you know and what you're unsure about
- Be concise but complete
- Don't mention "the context" - just answer naturally as if this is your knowledge

Answer:"""

        # Get response from Cohere
        chat_response = co.chat(
            model="command-r-plus-08-2024", 
            message=prompt,
            temperature=0.3  # Lower temperature for more factual responses
        )

        answer_text = getattr(chat_response, "text", str(chat_response)).strip()

        return {
            "answer": answer_text,
            "source": "learned_knowledge",
            "confidence": best_score,
            "matches_found": len(matches),
            "success": True
        }

    except Exception as e:
        print(f"❌ Error during ask: {e}")
        traceback.print_exc()
        return {
            "error": f"Failed to process question: {str(e)}",
            "success": False
        }

# -------------------------------
# 9️⃣ Forget endpoint - NEW
# -------------------------------
@app.post("/forget")
async def forget(data: ForgetRequest):
    """
    Remove learned knowledge.
    Can forget specific topics or everything.
    """
    try:
        if data.forget_all:
            # Delete all vectors
            stats = index.describe_index_stats()
            
            # Pinecone requires deleting by namespace or IDs
            # For simplicity, recreate the index
            print("🗑️  Deleting all knowledge...")
            pc.delete_index(index_name)
            time.sleep(2)
            pc.create_index(
                name=index_name,
                dimension=expected_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            
            return {
                "success": True,
                "message": "All knowledge has been forgotten",
                "items_deleted": "all"
            }
        
        elif data.query:
            # Find and delete specific content
            q_embed = get_embedding(data.query, input_type="search_query")
            results = index.query(
                vector=q_embed,
                top_k=10,
                include_metadata=True
            )
            
            matches = getattr(results, "matches", []) or results.get("matches", [])
            
            if not matches:
                return {
                    "success": True,
                    "message": "No matching knowledge found to forget",
                    "items_deleted": 0
                }
            
            # Delete highly relevant matches (>70% similarity)
            ids_to_delete = []
            for match in matches:
                score = match.get("score", 0) if isinstance(match, dict) else getattr(match, "score", 0)
                if score > 0.7:
                    match_id = match.get("id") if isinstance(match, dict) else getattr(match, "id")
                    ids_to_delete.append(match_id)
            
            if ids_to_delete:
                index.delete(ids=ids_to_delete)
                return {
                    "success": True,
                    "message": f"Forgot {len(ids_to_delete)} related knowledge items",
                    "items_deleted": len(ids_to_delete)
                }
            else:
                return {
                    "success": True,
                    "message": "No highly relevant items found to forget",
                    "items_deleted": 0
                }
        
        else:
            return {
                "error": "Please specify either 'query' to forget specific content or 'forget_all': true",
                "success": False
            }
    
    except Exception as e:
        print(f"❌ Error during forget: {e}")
        return {
            "error": f"Failed to forget: {str(e)}",
            "success": False
        }

# -------------------------------
# 🔟 Stats endpoint - NEW
# -------------------------------
@app.get("/stats")
async def get_stats():
    """Get statistics about learned knowledge"""
    try:
        stats = index.describe_index_stats()
        total_vectors = stats.total_vector_count if hasattr(stats, 'total_vector_count') else stats.get('total_vector_count', 0)
        
        return {
            "success": True,
            "total_knowledge_items": total_vectors,
            "index_name": index_name,
            "dimension": expected_dim,
            "status": "ready"
        }
    except Exception as e:
        return {
            "error": str(e),
            "success": False
        }

# -------------------------------
# 1️⃣1️⃣ List learned content - NEW
# -------------------------------
@app.get("/list")
async def list_knowledge():
    """
    List some examples of learned content.
    Note: This is a sample, not exhaustive.
    """
    try:
        # Query with a generic embedding to get some results
        sample_embed = get_embedding("knowledge information", input_type="search_query")
        results = index.query(
            vector=sample_embed,
            top_k=10,
            include_metadata=True
        )
        
        matches = getattr(results, "matches", []) or results.get("matches", [])
        
        items = []
        for match in matches:
            metadata = match.get("metadata", {}) if isinstance(match, dict) else getattr(match, "metadata", {})
            items.append({
                "preview": metadata.get("text", "")[:100] + "...",
                "length": metadata.get("length", 0),
                "timestamp": metadata.get("timestamp", "unknown")
            })
        
        return {
            "success": True,
            "sample_count": len(items),
            "items": items,
            "note": "This is a sample of learned content, not complete list"
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "success": False
        }

# -------------------------------
# 1️⃣2️⃣ Root endpoint with HTML interface
# -------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve a simple test interface"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Learning System</title>
        <style>
            body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #333; }
            .section { margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
            textarea { width: 100%; min-height: 100px; padding: 10px; font-size: 14px; }
            button { background: #007bff; color: white; border: none; padding: 10px 20px; 
                     border-radius: 4px; cursor: pointer; margin: 5px; }
            button:hover { background: #0056b3; }
            .response { margin-top: 15px; padding: 15px; background: #f8f9fa; 
                       border-radius: 4px; white-space: pre-wrap; }
            .error { background: #f8d7da; color: #721c24; }
            .success { background: #d4edda; color: #155724; }
        </style>
    </head>
    <body>
        <h1>🧠 RAG Learning System</h1>
        
        <div class="section">
            <h2>📚 Teach Me</h2>
            <textarea id="learnText" placeholder="Enter information to teach me..."></textarea>
            <button onclick="learn()">Learn</button>
            <div id="learnResponse" class="response" style="display:none;"></div>
        </div>
        
        <div class="section">
            <h2>❓ Ask Me</h2>
            <textarea id="askText" placeholder="Ask me a question..."></textarea>
            <button onclick="ask()">Ask</button>
            <div id="askResponse" class="response" style="display:none;"></div>
        </div>
        
        <div class="section">
            <h2>🗑️ Forget</h2>
            <textarea id="forgetText" placeholder="What should I forget? (leave empty to forget everything)"></textarea>
            <button onclick="forgetSpecific()">Forget This</button>
            <button onclick="forgetAll()" style="background: #dc3545;">Forget Everything</button>
            <div id="forgetResponse" class="response" style="display:none;"></div>
        </div>
        
        <div class="section">
            <h2>📊 Stats</h2>
            <button onclick="getStats()">Get Stats</button>
            <button onclick="listKnowledge()">List Knowledge</button>
            <div id="statsResponse" class="response" style="display:none;"></div>
        </div>

        <script>
            async function learn() {
                const text = document.getElementById('learnText').value;
                const resp = document.getElementById('learnResponse');
                resp.style.display = 'block';
                resp.className = 'response';
                resp.textContent = 'Learning...';
                
                try {
                    const res = await fetch('/learn', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({text})
                    });
                    const data = await res.json();
                    resp.className = data.success ? 'response success' : 'response error';
                    resp.textContent = JSON.stringify(data, null, 2);
                } catch(e) {
                    resp.className = 'response error';
                    resp.textContent = 'Error: ' + e.message;
                }
            }
            
            async function ask() {
                const text = document.getElementById('askText').value;
                const resp = document.getElementById('askResponse');
                resp.style.display = 'block';
                resp.className = 'response';
                resp.textContent = 'Thinking...';
                
                try {
                    const res = await fetch('/ask', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({text})
                    });
                    const data = await res.json();
                    resp.className = data.success ? 'response success' : 'response error';
                    resp.textContent = data.answer || JSON.stringify(data, null, 2);
                } catch(e) {
                    resp.className = 'response error';
                    resp.textContent = 'Error: ' + e.message;
                }
            }
            
            async function forgetSpecific() {
                const query = document.getElementById('forgetText').value;
                if (!query) {
                    alert('Please enter what to forget, or use "Forget Everything"');
                    return;
                }
                const resp = document.getElementById('forgetResponse');
                resp.style.display = 'block';
                resp.textContent = 'Forgetting...';
                
                try {
                    const res = await fetch('/forget', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({query})
                    });
                    const data = await res.json();
                    resp.className = data.success ? 'response success' : 'response error';
                    resp.textContent = JSON.stringify(data, null, 2);
                } catch(e) {
                    resp.className = 'response error';
                    resp.textContent = 'Error: ' + e.message;
                }
            }
            
            async function forgetAll() {
                if (!confirm('Are you sure you want to forget EVERYTHING?')) return;
                
                const resp = document.getElementById('forgetResponse');
                resp.style.display = 'block';
                resp.textContent = 'Forgetting everything...';
                
                try {
                    const res = await fetch('/forget', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({forget_all: true})
                    });
                    const data = await res.json();
                    resp.className = data.success ? 'response success' : 'response error';
                    resp.textContent = JSON.stringify(data, null, 2);
                } catch(e) {
                    resp.className = 'response error';
                    resp.textContent = 'Error: ' + e.message;
                }
            }
            
            async function getStats() {
                const resp = document.getElementById('statsResponse');
                resp.style.display = 'block';
                resp.textContent = 'Loading...';
                
                try {
                    const res = await fetch('/stats');
                    const data = await res.json();
                    resp.className = data.success ? 'response success' : 'response error';
                    resp.textContent = JSON.stringify(data, null, 2);
                } catch(e) {
                    resp.className = 'response error';
                    resp.textContent = 'Error: ' + e.message;
                }
            }
            
            async function listKnowledge() {
                const resp = document.getElementById('statsResponse');
                resp.style.display = 'block';
                resp.textContent = 'Loading...';
                
                try {
                    const res = await fetch('/list');
                    const data = await res.json();
                    resp.className = data.success ? 'response success' : 'response error';
                    resp.textContent = JSON.stringify(data, null, 2);
                } catch(e) {
                    resp.className = 'response error';
                    resp.textContent = 'Error: ' + e.message;
                }
            }
        </script>
    </body>
    </html>
    """
    return html

# -------------------------------
# ✅ Health check
# -------------------------------
@app.get("/health")
async def health():
    return {"status": "healthy", "service": "RAG Learning System"}

# -------------------------------
# Run: python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# -------------------------------
