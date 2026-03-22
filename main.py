from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import os, traceback, uuid, time, re
from datetime import datetime
from dotenv import load_dotenv
import cohere
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import httpx
import urllib.parse

# ── DynamoDB (chat sync) ──────────────────────────────────────────────────────
import boto3
from boto3.dynamodb.conditions import Key

# -------------------------------
# 1️⃣ Load environment variables
# -------------------------------
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Qdrant Cloud — set these 2 vars on Render dashboard
QDRANT_URL    = os.getenv("QDRANT_URL",    "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = "personal_knowledge"

# DynamoDB — set these on Render dashboard (Environment tab)
AWS_REGION   = os.getenv("AWS_REGION",   "us-east-1")
DYNAMO_TABLE = os.getenv("DYNAMO_TABLE", "personal_ai_chat")
USER_ID      = os.getenv("USER_ID",      "rahul_personal_ai")

# Web search — free public APIs, no key needed
WEB_SEARCH_ENABLED = os.getenv("WEB_SEARCH_ENABLED", "true").lower() == "true"

if not COHERE_API_KEY:
    raise ValueError("❌ Missing COHERE_API_KEY! Check .env file")

co = cohere.Client(COHERE_API_KEY)

# -------------------------------
# 2️⃣ Initialize Qdrant
# -------------------------------
def _qdrant() -> QdrantClient:
    """Return a Qdrant client connected to Qdrant Cloud."""
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def _init_qdrant():
    """Create the knowledge collection if it does not exist."""
    try:
        client = _qdrant()
        existing = [c.name for c in client.get_collections().collections]
        if QDRANT_COLLECTION not in existing:
            client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )
            print(f"✅ Qdrant collection '{QDRANT_COLLECTION}' created")
        else:
            print(f"✅ Qdrant collection '{QDRANT_COLLECTION}' ready")
    except Exception as e:
        print(f"⚠️ Qdrant init: {e}")

_init_qdrant()

# ── DynamoDB lazy init ────────────────────────────────────────────────────────
_dynamo_resource = None

def _table():
    global _dynamo_resource
    if _dynamo_resource is None:
        _dynamo_resource = boto3.resource("dynamodb", region_name=AWS_REGION)
    return _dynamo_resource.Table(DYNAMO_TABLE)

USER_PK = f"USER#{USER_ID}"

def _conv_key(conv_id):         return f"CONV#{conv_id}"
def _msg_key(conv_id, ts, mid): return f"MSG#{conv_id}#{ts:020d}#{mid}"
def _msg_prefix(conv_id):       return f"MSG#{conv_id}#"

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
    chunk_size: Optional[int] = 400

class AskRequest(BaseModel):
    text: str
    top_k: Optional[int] = 5
    use_general_knowledge: Optional[bool] = True

class ForgetRequest(BaseModel):
    query: Optional[str] = None
    forget_all: Optional[bool] = False

# ── DynamoDB chat models ──────────────────────────────────────────────────────
class ConvIn(BaseModel):
    id        : str
    title     : str
    createdAt : int
    updatedAt : int

class MsgIn(BaseModel):
    id              : str
    conversationId  : str
    role            : str
    content         : str
    timestamp       : int
    source          : Optional[str] = None
    isLoading       : bool          = False
    editGroupId     : Optional[str] = None
    editIndex       : int           = 0
    editTotal       : int           = 1
    activeEditIndex : int           = 0

# ══════════════════════════════════════════════════════════════════════════════
# 🌐  WEB SEARCH HELPERS  (DuckDuckGo + Wikipedia — no API key needed)
# ══════════════════════════════════════════════════════════════════════════════

async def _ddg_search(query: str, max_results: int = 5) -> List[dict]:
    """DuckDuckGo Instant Answer API — free, no key, no sign-up."""
    results = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; PersonalAI/1.0)"}
    try:
        params = {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get("https://api.duckduckgo.com/", params=params, headers=headers)
            d = r.json()
        if d.get("AbstractText"):
            results.append({
                "title"  : d.get("Heading", query),
                "snippet": d["AbstractText"][:500],
                "url"    : d.get("AbstractURL", ""),
                "source" : "DuckDuckGo/Wikipedia"
            })
        for topic in d.get("RelatedTopics", [])[:3]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title"  : topic.get("Text", "")[:80],
                    "snippet": topic.get("Text", "")[:300],
                    "url"    : topic.get("FirstURL", ""),
                    "source" : "DuckDuckGo"
                })
        for row in d.get("Infobox", {}).get("content", [])[:3]:
            if row.get("value"):
                results.append({
                    "title"  : row.get("label", ""),
                    "snippet": f"{row.get('label','')}: {row.get('value','')}",
                    "url"    : d.get("AbstractURL", ""),
                    "source" : "DuckDuckGo Infobox"
                })
    except Exception as e:
        print(f"⚠️ DDG search failed: {e}")
    return results[:max_results]


async def _wikipedia_search(query: str, sentences: int = 4) -> dict | None:
    """Wikipedia REST API — free, reliable for factual topics."""
    try:
        base = "https://en.wikipedia.org/w/api.php"
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(base, params={
                "action": "query", "list": "search", "srsearch": query,
                "srlimit": 1, "format": "json", "utf8": 1
            })
            hits = r.json().get("query", {}).get("search", [])
        if not hits:
            return None
        title = hits[0]["title"]
        async with httpx.AsyncClient(timeout=8) as client:
            r2 = await client.get(base, params={
                "action": "query", "prop": "extracts", "exintro": True,
                "exsentences": sentences, "explaintext": True,
                "titles": title, "format": "json", "utf8": 1
            })
            pages = r2.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()))
        extract = page.get("extract", "").strip()
        if not extract:
            return None
        return {
            "title"  : title,
            "snippet": extract[:600],
            "url"    : f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
            "source" : "Wikipedia"
        }
    except Exception as e:
        print(f"⚠️ Wikipedia search failed: {e}")
        return None


async def web_search(query: str, include_wiki: bool = True) -> str:
    """
    Run DDG + Wikipedia concurrently, return a formatted context string.
    Returns empty string on failure so the LLM still answers from training.
    """
    if not WEB_SEARCH_ENABLED:
        return ""
    print(f"🌐 Web search: {query!r}")
    import asyncio
    tasks = [_ddg_search(query, max_results=4)]
    if include_wiki:
        tasks.append(_wikipedia_search(query, sentences=5))
    results_list = await asyncio.gather(*tasks, return_exceptions=True)
    ddg_results = results_list[0] if not isinstance(results_list[0], Exception) else []
    wiki_result = (results_list[1]
                   if len(results_list) > 1 and not isinstance(results_list[1], Exception)
                   else None)
    all_results = []
    if wiki_result:
        all_results.append(wiki_result)
    for r in ddg_results:
        if wiki_result and r["snippet"][:80] in wiki_result["snippet"]:
            continue
        all_results.append(r)
    if not all_results:
        print("⚠️ No web results found")
        return ""
    lines = ["[Web Search Results]"]
    for i, r in enumerate(all_results[:5], 1):
        lines.append(f"\n[{i}] {r['title']} ({r['source']})")
        lines.append(f"    {r['snippet']}")
        if r.get("url"):
            lines.append(f"    URL: {r['url']}")
    print(f"✅ Web search: {len(all_results)} results")
    return "\n".join(lines)


def _needs_web_search(question: str) -> bool:
    """Return True if the question likely needs live/current data."""
    q = question.lower()
    triggers = [
        "today", "right now", "currently", "latest", "recent", "news",
        "price", "stock", "weather", "score", "result", "winner",
        "who is the", "who won", "what happened", "2024", "2025", "2026",
        "pm of", "cm of", "president of", "ceo of", "prime minister",
        "chief minister", "governor", "minister", "election",
        "how much does", "how much is", "what is the current",
        "live", "update", "just", "this week", "this month", "this year",
        "trending", "new", "released", "launched", "announced"
    ]
    if any(t in q for t in triggers):
        return True
    if re.match(r"^(who|what|when|where|which|how much|how many)\s", q):
        return True
    return False

# -------------------------------
# 6️⃣ Helper functions
# -------------------------------
def get_embedding(text: str, input_type: str = "search_document"):
    try:
        response = co.embed(
            texts=[text],
            model="embed-english-v3.0",
            input_type=input_type,
            truncate="END"
        )
        return response.embeddings[0]
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        raise

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            last_period = chunk.rfind(".")
            last_newline = chunk.rfind("\n")
            break_point = max(last_period, last_newline)
            if break_point > chunk_size * 0.5:
                chunk = chunk[:break_point + 1]
                end = start + break_point + 1
        chunks.append(chunk.strip())
        start = end - overlap
    return chunks

# -------------------------------
# 7️⃣ Learn endpoint
# -------------------------------
@app.post("/learn")
async def learn(data: LearnRequest):
    text = data.text.strip()
    if not text:
        return {"error": "No text provided", "success": False}
    if len(text) < 10:
        return {"error": "Text too short - provide at least 10 characters", "success": False}

    chunks = chunk_text(text, chunk_size=data.chunk_size)
    new_count = updated_count = skipped_count = 0
    print(f"\n📚 Learning {len(chunks)} chunks...")

    try:
        client = _qdrant()
        for idx, chunk in enumerate(chunks):
            try:
                embedding = get_embedding(chunk, input_type="search_document")

                # Near-duplicate check (cosine similarity > 0.95)
                hits = client.search(
                    collection_name=QDRANT_COLLECTION,
                    query_vector=embedding,
                    limit=1,
                    with_payload=True,
                )
                if hits and hits[0].score > 0.95:
                    old_id   = hits[0].id
                    old_text = hits[0].payload.get("content", "")
                    if len(chunk) > len(old_text):
                        # Update in place — Qdrant upsert replaces by ID
                        client.upsert(
                            collection_name=QDRANT_COLLECTION,
                            points=[PointStruct(
                                id=old_id,
                                vector=embedding,
                                payload={"content": chunk, "timestamp": datetime.now().isoformat(), "length": len(chunk)}
                            )]
                        )
                        updated_count += 1
                        print(f"  🔄 Chunk {idx+1}: Updated (score:{hits[0].score:.3f})")
                    else:
                        skipped_count += 1
                        print(f"  ⏭️  Chunk {idx+1}: Skipped (score:{hits[0].score:.3f})")
                else:
                    client.upsert(
                        collection_name=QDRANT_COLLECTION,
                        points=[PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embedding,
                            payload={"content": chunk, "timestamp": datetime.now().isoformat(), "length": len(chunk)}
                        )]
                    )
                    new_count += 1
                    print(f"  ✅ Chunk {idx+1}: Stored")
            except Exception as e:
                print(f"  ❌ Chunk {idx+1}: {e}"); continue

        total = client.get_collection(QDRANT_COLLECTION).points_count
    except Exception as e:
        return {"error": f"Qdrant error: {str(e)}", "success": False}

    return {
        "success": True,
        "chunks_processed": len(chunks),
        "chunks_stored": new_count,
        "chunks_updated": updated_count,
        "chunks_skipped": skipped_count,
        "total_knowledge_items": total,
        "message": f"Learned successfully! {new_count} new, {updated_count} updated, {skipped_count} skipped (duplicates)"
    }

# -------------------------------
# 8️⃣ Ask endpoint - HYBRID + WEB SEARCH
# -------------------------------
@app.post("/ask")
async def ask(data: AskRequest):
    question = data.text.strip()
    if not question:
        return {"error": "No question provided", "success": False}
    print(f"\n❓ Question: {question}")

    try:
        # ── Step 1: Personal knowledge (Qdrant) ─────────────────────────────
        q_embed = get_embedding(question, input_type="search_query")
        client  = _qdrant()
        hits    = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=q_embed,
            limit=data.top_k,
            with_payload=True,
        )

        best_score   = hits[0].score if hits else 0
        has_personal = bool(hits) and best_score > 0.25
        print(f"🔍 Personal: {len(hits)} matches, best:{best_score:.3f}, relevant:{has_personal}")

        # ── Step 2: Live web search if needed ────────────────────────────────
        needs_web   = _needs_web_search(question)
        web_context = await web_search(question) if needs_web else ""

        # ── Step 3: Build prompt + call Cohere ───────────────────────────────
        if has_personal:
            personal_context = "\n\n".join(
                f"[Personal Knowledge {i+1}, relevance:{h.score:.2f}]\n{h.payload.get('content','')}"
                for i, h in enumerate(hits) if h.score > 0.2
            )
            web_section = f"\n\nLive Web Search Results:\n{web_context}" if web_context else ""
            prompt = f"""You are a helpful personal AI assistant. Answer using all available context.

Question: {question}

Personal Knowledge (from user's stored data):
{personal_context}{web_section}

Instructions:
- Prioritise personal knowledge when directly relevant
- Use web search results for current/factual data
- Combine both sources for the most complete answer
- Be conversational and helpful

Answer:"""
            source = "personal_knowledge"
            note   = "Answer based on your learned data" + (" + live web search" if web_context else "")

        elif data.use_general_knowledge:
            web_section = f"\nLive Web Search Results:\n{web_context}\n" if web_context else ""
            prompt = f"""You are a helpful AI assistant with access to live web data.

Question: {question}
{web_section}
Instructions:
- {"Use the web search results above to give an accurate up-to-date answer" if web_context else "Answer from your training knowledge"}
- Be conversational and concise

Answer:"""
            source = "web_search" if web_context else "general_knowledge"
            note   = "Answer from live web search" if web_context else "Answer from general AI knowledge (not your personal data)"

        else:
            return {"answer": "I don't have relevant information in my learned knowledge base about this topic. You can enable general knowledge mode or teach me about it using /learn.",
                    "source": "none", "confidence": best_score, "success": True}

        chat_response = co.chat(
            model="command-r-plus-08-2024",
            message=prompt,
            temperature=0.4 if web_context else 0.5,
        )
        answer_text = getattr(chat_response, "text", str(chat_response)).strip()
        return {
            "answer"       : answer_text,
            "source"       : source,
            "confidence"   : best_score,
            "matches_found": len(hits),
            "web_searched" : bool(web_context),
            "success"      : True,
            "note"         : note,
        }

    except Exception as e:
        print(f"❌ Ask error: {e}"); traceback.print_exc()
        return {"error": f"Failed to process question: {str(e)}", "success": False}

# -------------------------------
# 9️⃣ Forget endpoint
# -------------------------------
@app.post("/forget")
async def forget(data: ForgetRequest):
    try:
        client = _qdrant()
        if data.forget_all:
            # Delete and recreate the collection
            client.delete_collection(QDRANT_COLLECTION)
            client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )
            return {"success": True, "message": "All knowledge has been forgotten", "items_deleted": "all"}
        elif data.query:
            q_embed = get_embedding(data.query, input_type="search_query")
            hits    = client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=q_embed,
                limit=10,
                with_payload=False,
            )
            ids_to_delete = [h.id for h in hits if h.score > 0.7]
            if ids_to_delete:
                client.delete(
                    collection_name=QDRANT_COLLECTION,
                    points_selector=ids_to_delete,
                )
                return {"success": True, "message": f"Forgot {len(ids_to_delete)} related knowledge items", "items_deleted": len(ids_to_delete)}
            return {"success": True, "message": "No highly relevant items found to forget", "items_deleted": 0}
        else:
            return {"error": "Please specify either 'query' to forget specific content or 'forget_all': true", "success": False}
    except Exception as e:
        print(f"❌ Error during forget: {e}")
        return {"error": f"Failed to forget: {str(e)}", "success": False}

# -------------------------------
# 🔟 Stats endpoint
# -------------------------------
@app.get("/stats")
async def get_stats():
    try:
        client = _qdrant()
        info   = client.get_collection(QDRANT_COLLECTION)
        return {"success": True,
                "total_knowledge_items": info.points_count,
                "storage": "Qdrant Cloud",
                "status" : "ready"}
    except Exception as e:
        return {"error": str(e), "success": False}

# -------------------------------
# 1️⃣1️⃣ List learned content
# -------------------------------
@app.get("/list")
async def list_knowledge():
    try:
        client  = _qdrant()
        results, _ = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=10,
            with_payload=True,
            with_vectors=False,
        )
        items = [{
            "preview"  : r.payload.get("content", "")[:100] + "...",
            "length"   : r.payload.get("length", 0),
            "timestamp": r.payload.get("timestamp", "unknown"),
        } for r in results]
        return {"success": True, "sample_count": len(items), "items": items,
                "note": "This is a sample of learned content, not complete list"}
    except Exception as e:
        return {"error": str(e), "success": False}

# -------------------------------
# 1️⃣2️⃣ Root endpoint with HTML interface
# -------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Learning System</title>
        <style>
            body { font-family: Arial; max-width: 900px; margin: 50px auto; padding: 20px; }
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
            .info-box { background: #d1ecf1; border-left: 4px solid #0c5460; padding: 15px; margin: 20px 0; }
            .source-badge { display: inline-block; padding: 5px 10px; border-radius: 12px; 
                           font-size: 12px; font-weight: bold; margin: 5px 0; }
            .badge-personal { background: #28a745; color: white; }
            .badge-general { background: #17a2b8; color: white; }
            .badge-none { background: #6c757d; color: white; }
        </style>
    </head>
    <body>
        <h1>&#x1F9E0; RAG Learning System</h1>
        <div class="info-box">
            <strong>&#x1F4A1; How it works:</strong><br>
            &bull; <strong>Personal Mode:</strong> I'll answer from what you teach me via /learn<br>
            &bull; <strong>General Mode:</strong> I'll answer worldly questions using AI knowledge<br>
            &bull; <strong>Hybrid:</strong> I combine both for the best answers!
        </div>
        <div class="section">
            <h2>&#x1F4DA; Teach Me (Personal Knowledge)</h2>
            <textarea id="learnText" placeholder="Enter information to teach me..."></textarea>
            <button onclick="learn()">Learn</button>
            <div id="learnResponse" class="response" style="display:none;"></div>
        </div>
        <div class="section">
            <h2>&#x2753; Ask Me Anything</h2>
            <textarea id="askText" placeholder="Ask me anything..."></textarea>
            <button onclick="ask()">Ask</button>
            <div id="askResponse" class="response" style="display:none;"></div>
        </div>
        <div class="section">
            <h2>&#x1F5D1;&#xFE0F; Forget</h2>
            <textarea id="forgetText" placeholder="What should I forget? (leave empty to forget everything)"></textarea>
            <button onclick="forgetSpecific()">Forget This</button>
            <button onclick="forgetAll()" style="background: #dc3545;">Forget Everything</button>
            <div id="forgetResponse" class="response" style="display:none;"></div>
        </div>
        <div class="section">
            <h2>&#x1F4CA; Stats &amp; Management</h2>
            <button onclick="getStats()">Get Stats</button>
            <button onclick="listKnowledge()">List Knowledge</button>
            <div id="statsResponse" class="response" style="display:none;"></div>
        </div>
        <script>
            async function learn() {
                const text = document.getElementById('learnText').value;
                const resp = document.getElementById('learnResponse');
                resp.style.display = 'block'; resp.className = 'response'; resp.textContent = 'Learning...';
                try {
                    const res = await fetch('/learn', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
                    const data = await res.json();
                    resp.className = data.success ? 'response success' : 'response error';
                    resp.textContent = JSON.stringify(data, null, 2);
                } catch(e) { resp.className='response error'; resp.textContent='Error: '+e.message; }
            }
            async function ask() {
                const text = document.getElementById('askText').value;
                const resp = document.getElementById('askResponse');
                resp.style.display='block'; resp.className='response'; resp.textContent='Thinking...';
                try {
                    const res = await fetch('/ask', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,use_general_knowledge:true})});
                    const data = await res.json();
                    resp.className = data.success ? 'response success' : 'response error';
                    let badgeClass='badge-none', badgeText='No Data';
                    if(data.source==='personal_knowledge'){badgeClass='badge-personal';badgeText='Personal Knowledge';}
                    else if(data.source==='web_search'){badgeClass='badge-general';badgeText='Live Web Search';}
                    else if(data.source==='general_knowledge'){badgeClass='badge-general';badgeText='General Knowledge';}
                    resp.innerHTML = '<div class="source-badge '+badgeClass+'">'+badgeText+'</div>'+(data.web_searched?'<div style="font-size:11px;color:#555;margin:2px 0">&#x1F310; Searched the web</div>':'')+
                        '<div style="margin-top:10px"><strong>Answer:</strong></div><div style="margin-top:10px">'+(data.answer||'No answer')+'</div>'+
                        (data.confidence?'<div style="margin-top:10px;font-size:12px;color:#666">Confidence: '+(data.confidence*100).toFixed(1)+'%</div>':'')+
                        (data.note?'<div style="margin-top:10px;font-size:12px;font-style:italic;color:#666">'+data.note+'</div>':'');
                } catch(e) { resp.className='response error'; resp.textContent='Error: '+e.message; }
            }
            async function forgetSpecific() {
                const query = document.getElementById('forgetText').value;
                if(!query){alert('Please enter what to forget, or use "Forget Everything"');return;}
                const resp=document.getElementById('forgetResponse');
                resp.style.display='block'; resp.textContent='Forgetting...';
                try {
                    const res=await fetch('/forget',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query})});
                    const data=await res.json();
                    resp.className=data.success?'response success':'response error';
                    resp.textContent=JSON.stringify(data,null,2);
                } catch(e){resp.className='response error';resp.textContent='Error: '+e.message;}
            }
            async function forgetAll() {
                if(!confirm('Are you sure you want to forget EVERYTHING? This cannot be undone!'))return;
                const resp=document.getElementById('forgetResponse');
                resp.style.display='block'; resp.textContent='Forgetting everything...';
                try {
                    const res=await fetch('/forget',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({forget_all:true})});
                    const data=await res.json();
                    resp.className=data.success?'response success':'response error';
                    resp.textContent=JSON.stringify(data,null,2);
                } catch(e){resp.className='response error';resp.textContent='Error: '+e.message;}
            }
            async function getStats() {
                const resp=document.getElementById('statsResponse');
                resp.style.display='block'; resp.textContent='Loading...';
                try{const res=await fetch('/stats');const data=await res.json();resp.className=data.success?'response success':'response error';resp.textContent=JSON.stringify(data,null,2);}
                catch(e){resp.className='response error';resp.textContent='Error: '+e.message;}
            }
            async function listKnowledge() {
                const resp=document.getElementById('statsResponse');
                resp.style.display='block'; resp.textContent='Loading...';
                try{const res=await fetch('/list');const data=await res.json();resp.className=data.success?'response success':'response error';resp.textContent=JSON.stringify(data,null,2);}
                catch(e){resp.className='response error';resp.textContent='Error: '+e.message;}
            }
        </script>
    </body>
    </html>
    """
    return html

# -------------------------------
# ✅ Health check
# -------------------------------
@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"ok": True, "service": "Personal AI Backend"}

# ══════════════════════════════════════════════════════════════════════════════
# 🗄️  DYNAMODB CHAT SYNC ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/convs")
async def list_convs():
    """List all conversations ordered by most-recently-updated first."""
    try:
        resp  = _table().query(
            IndexName              = "gsi1",
            KeyConditionExpression = Key("gsi1pk").eq(USER_PK),
            ScanIndexForward       = False,
        )
        convs = []
        for item in resp.get("Items", []):
            if str(item.get("sk", "")).startswith("CONV#"):
                convs.append({
                    "id"       : item["convId"],
                    "title"    : item.get("title",     "Chat"),
                    "createdAt": int(item.get("createdAt", 0)),
                    "updatedAt": int(item.get("gsi1sk",    0)),
                })
        convs.sort(key=lambda c: c["updatedAt"], reverse=True)
        return {"success": True, "conversations": convs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conv/{conv_id}")
async def get_conv(conv_id: str):
    """Get a single conversation by ID (used by the share page)."""
    try:
        resp = _table().get_item(Key={"pk": USER_PK, "sk": _conv_key(conv_id)})
        item = resp.get("Item")
        if not item:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {
            "success"  : True,
            "id"       : item["convId"],
            "title"    : item.get("title",     "Chat"),
            "createdAt": int(item.get("createdAt", 0)),
            "updatedAt": int(item.get("gsi1sk",    0)),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/conv")
async def upsert_conv(body: ConvIn):
    """Create or update a conversation."""
    try:
        if body.title:
            _table().put_item(Item={
                "pk"        : USER_PK,
                "sk"        : _conv_key(body.id),
                "gsi1pk"    : USER_PK,
                "gsi1sk"    : str(body.updatedAt),
                "convId"    : body.id,
                "title"     : body.title,
                "createdAt" : body.createdAt,
            })
        else:
            _table().update_item(
                Key={"pk": USER_PK, "sk": _conv_key(body.id)},
                UpdateExpression="SET gsi1sk = :ts",
                ExpressionAttributeValues={":ts": str(body.updatedAt)}
            )
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/conv/{conv_id}")
async def delete_conv(conv_id: str):
    """Delete a conversation and all its messages in one batch."""
    try:
        resp = _table().query(
            KeyConditionExpression=(
                Key("pk").eq(USER_PK) &
                Key("sk").begins_with(_msg_prefix(conv_id))
            )
        )
        with _table().batch_writer() as batch:
            for item in resp.get("Items", []):
                batch.delete_item(Key={"pk": item["pk"], "sk": item["sk"]})
            batch.delete_item(Key={"pk": USER_PK, "sk": _conv_key(conv_id)})
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/msgs/{conv_id}")
async def list_msgs(conv_id: str, since: Optional[int] = None):
    """List messages for a conversation, oldest-first."""
    try:
        resp  = _table().query(
            KeyConditionExpression=(
                Key("pk").eq(USER_PK) &
                Key("sk").begins_with(_msg_prefix(conv_id))
            ),
            ScanIndexForward=True
        )
        items = resp.get("Items", [])
        if since:
            items = [i for i in items if int(i.get("timestamp", 0)) > since]
        msgs = []
        for item in items:
            msgs.append({
                "id"             : item["msgId"],
                "conversationId" : item.get("convId",             conv_id),
                "role"           : item.get("role",                "ai"),
                "content"        : item.get("content",             ""),
                "timestamp"      : int(item.get("timestamp",       0)),
                "source"         : item.get("source")              or None,
                "isLoading"      : bool(item.get("isLoading",      False)),
                "editGroupId"    : item.get("editGroupId")         or None,
                "editIndex"      : int(item.get("editIndex",       0)),
                "editTotal"      : int(item.get("editTotal",       1)),
                "activeEditIndex": int(item.get("activeEditIndex", 0)),
            })
        return {"success": True, "messages": msgs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/msg")
async def upsert_msg(body: MsgIn):
    """Create or update a message, and refresh the conversation updatedAt."""
    try:
        _table().put_item(Item={
            "pk"             : USER_PK,
            "sk"             : _msg_key(body.conversationId, body.timestamp, body.id),
            "msgId"          : body.id,
            "convId"         : body.conversationId,
            "role"           : body.role,
            "content"        : body.content,
            "timestamp"      : body.timestamp,
            "source"         : body.source          or "",
            "isLoading"      : body.isLoading,
            "editGroupId"    : body.editGroupId      or "",
            "editIndex"      : body.editIndex,
            "editTotal"      : body.editTotal,
            "activeEditIndex": body.activeEditIndex,
        })
        now = int(time.time() * 1000)
        _table().update_item(
            Key={"pk": USER_PK, "sk": _conv_key(body.conversationId)},
            UpdateExpression="SET gsi1sk = :ts",
            ExpressionAttributeValues={":ts": str(now)}
        )
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/msg/{conv_id}/{msg_id}")
async def delete_msg(conv_id: str, msg_id: str):
    """Delete a specific message."""
    try:
        resp = _table().query(
            KeyConditionExpression=(
                Key("pk").eq(USER_PK) &
                Key("sk").begins_with(_msg_prefix(conv_id))
            )
        )
        for item in resp.get("Items", []):
            if item.get("msgId") == msg_id:
                _table().delete_item(Key={"pk": item["pk"], "sk": item["sk"]})
                return {"success": True}
        return {"success": False, "error": "Message not found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
async def search(q: str, wiki: bool = True):
    """Standalone web search. GET /search?q=your+query"""
    if not q:
        return {"error": "Provide ?q= query param", "success": False}
    try:
        context = await web_search(q, include_wiki=wiki)
        return {"success": True, "query": q, "context": context, "has_results": bool(context)}
    except Exception as e:
        return {"error": str(e), "success": False}


@app.post("/generate_title")
async def generate_title(data: dict):
    """Generate a short title for a conversation based on first user+AI messages."""
    try:
        user_text = data.get("user_text", "")
        ai_text   = data.get("ai_text", "")
        prompt = f"""Generate a short, descriptive title (max 5 words) for a conversation that starts with:

User: {user_text[:200]}
AI: {ai_text[:200]}

Rules:
- Max 5 words
- No quotes, no punctuation at the end
- Capture the main topic
- Examples: "Python list sorting help", "Travel plans Tokyo", "Recipe ideas pasta"

Title:"""
        response = co.chat(model="command-r-plus-08-2024", message=prompt, temperature=0.3)
        title = getattr(response, "text", "New Chat").strip().strip('"').strip("'")
        if len(title) > 50:
            title = title[:47] + "..."
        return {"success": True, "title": title}
    except Exception as e:
        return {"success": False, "title": "New Chat", "error": str(e)}


# -------------------------------
# Run: python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# -------------------------------
