# 🐛 DEBUG_RETRIEVAL.md - Vector Store Retrieval Debugging Guide

> **Your Debugging Companion**: A comprehensive guide to understand, diagnose, and fix vector store retrieval issues in the RAG MCQ system.

## 📋 Table of Contents
1. [Understanding the Problem](#understanding-the-problem)
2. [The Retrieval Flow](#the-retrieval-flow)
3. [Common Root Causes](#common-root-causes)
4. [Debugging Methodology](#debugging-methodology)
5. [Step-by-Step Debugging Process](#step-by-step-debugging-process)
6. [Do's and Don'ts](#dos-and-donts)
7. [Advanced Debugging Techniques](#advanced-debugging-techniques)
8. [Prevention Strategies](#prevention-strategies)

---

## 🎯 Understanding the Problem

### What's Happening?
When you get **"No relevant context found for topic: Machine Learning"**, it means:

1. **Vector Store is Empty**: No documents were properly embedded
2. **Search Mismatch**: Your query doesn't match embedded content
3. **Threshold Too High**: Similarity thresholds are filtering out results
4. **Embedding Issues**: Documents weren't processed correctly

### The Error Flow
```
User Request → MCQ Controller → Use Case → MCQ Service → Vector Store → EMPTY RESULTS → Error
```

---

## 🔄 The Retrieval Flow

### Complete Data Journey

```mermaid
graph LR
    A[PDF Upload] --> B[Document Processing]
    B --> C[Text Chunking]
    C --> D[Embedding Generation]
    D --> E[FAISS Storage]
    E --> F[Query Search]
    F --> G[Results Filtering]
    G --> H[Context Return]
```

### Key Components Involved

1. **Document Upload** (`/api/v1/documents/upload`)
   - File validation and storage
   - Background processing trigger

2. **Document Processing** (`Infrastructure/document_processing/pdf_processor.py`)
   - PDF text extraction
   - Content chunking

3. **Vector Store** (`Infrastructure/vector_stores/faiss_store.py`)
   - Embedding generation
   - FAISS index creation
   - Similarity search

4. **MCQ Generation** (`Domain/services/mcq_generation_service.py`)
   - Context retrieval
   - Question generation

---

## 🚨 Common Root Causes

### 1. Documents Not Uploaded/Processed ⚠️
**Symptoms:**
- Vector store returns 0 documents
- `is_initialized()` returns `False`
- No chunks in FAISS index

**Why it happens:**
```python
# Vector store is empty
await vector_store.get_document_count()  # Returns 0
```

### 2. Background Processing Not Complete ⏳
**Symptoms:**
- Upload succeeds but search fails
- Document count is 0 shortly after upload

**Why it happens:**
```python
# Upload triggers background processing
background_tasks.add_task(process_document_background, file_path)
# But search happens before processing completes
```

### 3. Query-Content Mismatch 🎯
**Symptoms:**
- Documents exist but no results for specific topics
- Works for some queries but not others

**Why it happens:**
```python
# Your query: "Machine Learning"
# Document content: "AI algorithms and neural networks..."
# Embedding similarity too low to match
```

### 4. Similarity Threshold Too Strict 📊
**Symptoms:**
- Search returns candidates but filters them out
- Lowering threshold fixes the issue

**Why it happens:**
```python
# In faiss_store.py
similarity_score = 1.0 / (1.0 + score)
if similarity_score >= similarity_threshold:  # 0.3 default
    # Document included
```

### 5. Embedding Model Issues 🤖
**Symptoms:**
- Documents processed but poor search quality
- Inconsistent results

**Why it happens:**
```python
# Vietnamese model for English content
embedding_model: "bkai-foundation-models/vietnamese-bi-encoder"
# May not work well for English queries
```

---

## 🔍 Debugging Methodology

### The TRACE Method

**T** - **Test** the basics first
**R** - **Review** the data flow
**A** - **Analyze** each component
**C** - **Check** configurations
**E** - **Experiment** with parameters

### Debug Priority Order
1. ✅ **Data Existence** (Is there any data?)
2. ✅ **Data Processing** (Was data processed correctly?)
3. ✅ **Search Execution** (Is search working?)
4. ✅ **Result Filtering** (Are results being filtered out?)
5. ✅ **Configuration** (Are settings correct?)

---

## 🛠️ Step-by-Step Debugging Process

### Step 1: Verify System Health
```bash
# Check if system is running
curl -X GET "http://localhost:8000/health"

# Expected response:
{
  "status": "healthy",
  "services": {
    "llm_service": true,
    "vector_store": true,     # Should be true
    "document_processor": true
  }
}
```

**🔍 Analysis:**
- If `vector_store: false` → Vector store initialization failed
- If all `true` → System is healthy, check data

### Step 2: Check Document Count
```bash
# Check if documents exist
curl -X GET "http://localhost:8000/api/v1/documents/stats"

# Expected response:
{
  "is_initialized": true,
  "embedding_model": "bkai-foundation-models/vietnamese-bi-encoder",
  "document_count": 42  # Should be > 0
}
```

**🔍 Analysis:**
- If `document_count: 0` → No documents processed
- If `is_initialized: false` → Vector store not ready
- If both are good → Check search logic

### Step 3: Upload Test Document (If Needed)
```bash
# Upload a test document
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@pdfs/Project_2_2_SpamClassifier_AIOTimeSeries.pdf"

# Expected response:
{
  "message": "Document uploaded successfully",
  "filename": "Project_2_2_SpamClassifier_AIOTimeSeries.pdf",
  "processing": "started in background"
}
```

**⏰ Wait Time:** Give it 30-60 seconds for processing to complete.

### Step 4: Re-check Document Count
```bash
# Wait 30 seconds, then check again
sleep 30
curl -X GET "http://localhost:8000/api/v1/documents/stats"

# Document count should now be > 0
```

### Step 5: Test with Document-Specific Query
```bash
# Use content that should exist in your PDF
curl -X POST "http://localhost:8000/api/v1/mcq/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Spam Classification",  # From your PDF title
    "question_type": "single_choice",
    "difficulty": "medium",
    "num_options": 4
  }'
```

**🔍 Analysis:**
- If this works → Query mismatch issue
- If still fails → Deeper investigation needed

### Step 6: Debug Search Parameters

**Create a test script** (`debug_search.py`):
```python
import asyncio
from Infrastructure.vector_stores.faiss_store import FAISSVectorStore

async def debug_search():
    # Initialize vector store
    vector_store = FAISSVectorStore(
        embedding_model="bkai-foundation-models/vietnamese-bi-encoder",
        chunk_size=500,
        diversity_threshold=0.7
    )

    print("🔍 Debugging Vector Store Search...")

    # Check initialization
    print(f"Is initialized: {vector_store.is_initialized()}")

    if not vector_store.is_initialized():
        print("❌ Vector store not initialized!")
        return

    # Check document count
    doc_count = await vector_store.get_document_count()
    print(f"Document count: {doc_count}")

    if doc_count == 0:
        print("❌ No documents in vector store!")
        return

    # Test different queries
    test_queries = [
        "Machine Learning",
        "Spam Classification",
        "Neural Networks",
        "Data Science",
        "Algorithm"
    ]

    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")

        # Test with different thresholds
        for threshold in [0.1, 0.2, 0.3, 0.4]:
            results = await vector_store.search_similar(
                query=query,
                k=5,
                similarity_threshold=threshold
            )
            print(f"  Threshold {threshold}: {len(results)} results")

            if results:
                print(f"    First result preview: {results[0].content[:100]}...")

if __name__ == "__main__":
    asyncio.run(debug_search())
```

**Run the test:**
```bash
python debug_search.py
```

### Step 7: Analyze Search Results

**Expected Output Analysis:**
```
🔍 Debugging Vector Store Search...
Is initialized: True
Document count: 15

🔍 Testing query: 'Machine Learning'
  Threshold 0.1: 3 results
  Threshold 0.2: 2 results
  Threshold 0.3: 0 results  ← PROBLEM HERE!
  Threshold 0.4: 0 results

🔍 Testing query: 'Spam Classification'
  Threshold 0.1: 5 results
  Threshold 0.2: 4 results
  Threshold 0.3: 3 results  ← This works!
```

**🔍 Analysis:**
- If "Machine Learning" gets 0 results at 0.3 threshold → Lower threshold
- If "Spam Classification" works → Query relevance issue
- If nothing works → Embedding/processing issue

---

## ✅ Do's and Don'ts

### ✅ DO's

#### 1. **Always Check Data First**
```bash
# Before debugging search, verify data exists
curl -X GET "http://localhost:8000/api/v1/documents/stats"
```

#### 2. **Use Gradual Debugging**
```python
# Start with loose parameters, then tighten
similarity_threshold = 0.1  # Very loose
k = 10  # More results
```

#### 3. **Test with Document Content**
```bash
# Use terms that definitely exist in your PDFs
# Check PDF content first: "Spam", "Classification", "AIO"
```

#### 4. **Wait for Background Processing**
```python
# Always wait after upload
import time
time.sleep(30)  # Give processing time
```

#### 5. **Log Everything During Debug**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# In your code
print(f"Search results: {len(results)}")
print(f"Query: {query}")
print(f"Threshold: {threshold}")
```

#### 6. **Use System Info Endpoint**
```bash
# Get comprehensive system state
curl -X GET "http://localhost:8000/api/v1/system/info"
```

### ❌ DON'Ts

#### 1. **Don't Skip Health Checks**
```bash
# ❌ Don't assume system is working
curl -X POST "http://localhost:8000/api/v1/mcq/generate" ...

# ✅ Always check health first
curl -X GET "http://localhost:8000/health"
```

#### 2. **Don't Use Vague Queries for Testing**
```bash
# ❌ Don't test with random topics
"topic": "Quantum Physics"  # Not in your PDF

# ✅ Use specific PDF content
"topic": "Spam Email Detection"  # From your PDF
```

#### 3. **Don't Ignore Thresholds**
```python
# ❌ Don't assume default thresholds work
similarity_threshold = 0.3  # Might be too strict

# ✅ Test with lower thresholds first
similarity_threshold = 0.1  # More permissive
```

#### 4. **Don't Test Immediately After Upload**
```bash
# ❌ Don't test right after upload
curl -X POST upload && curl -X POST generate

# ✅ Wait for processing
curl -X POST upload && sleep 30 && curl -X POST generate
```

#### 5. **Don't Mix Languages**
```python
# ❌ Don't use English queries with Vietnamese embeddings
embedding_model = "vietnamese-bi-encoder"
query = "Machine Learning"  # English

# ✅ Match language or use multilingual model
embedding_model = "all-MiniLM-L6-v2"  # Multilingual
```

---

## 🔬 Advanced Debugging Techniques

### 1. Deep Dive into FAISS Store

**Create detailed inspection script** (`inspect_vector_store.py`):
```python
import asyncio
import numpy as np
from Infrastructure.vector_stores.faiss_store import FAISSVectorStore

async def inspect_vector_store():
    vector_store = FAISSVectorStore()

    # Force initialization
    await vector_store._initialize()

    print("📊 Vector Store Inspection Report")
    print("=" * 50)

    # Check embeddings
    if vector_store.embeddings:
        print("✅ Embeddings initialized")
        # Test embedding generation
        test_text = "Machine Learning"
        embedding = vector_store.embeddings.embed_query(test_text)
        print(f"   Embedding dimension: {len(embedding)}")
        print(f"   Sample values: {embedding[:5]}")
    else:
        print("❌ Embeddings not initialized")

    # Check vector database
    if vector_store.vector_db:
        print("✅ Vector database exists")
        print(f"   Index total: {vector_store.vector_db.index.ntotal}")
        print(f"   Index dimension: {vector_store.vector_db.index.d}")

        # Get sample documents
        if vector_store.vector_db.index.ntotal > 0:
            # Search with very low threshold
            results = vector_store.vector_db.similarity_search_with_score(
                "test", k=3
            )
            print(f"   Sample search results: {len(results)}")
            for i, (doc, score) in enumerate(results):
                print(f"     {i+1}. Score: {score:.4f}")
                print(f"        Content: {doc.page_content[:100]}...")
    else:
        print("❌ Vector database not initialized")

if __name__ == "__main__":
    asyncio.run(inspect_vector_store())
```

### 2. Query-Document Similarity Analysis

```python
async def analyze_query_similarity():
    vector_store = FAISSVectorStore()
    await vector_store._initialize()

    if not vector_store.vector_db:
        print("No vector database found!")
        return

    # Test query
    query = "Machine Learning"

    # Get all documents with scores
    all_results = vector_store.vector_db.similarity_search_with_score(
        query, k=100  # Get many results
    )

    print(f"🔍 Similarity Analysis for: '{query}'")
    print("=" * 50)

    for i, (doc, distance) in enumerate(all_results[:10]):
        similarity = 1.0 / (1.0 + distance)
        print(f"{i+1}. Similarity: {similarity:.4f} | Distance: {distance:.4f}")
        print(f"   Content: {doc.page_content[:150]}...")
        print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
        print("-" * 30)
```

### 3. Real-time Monitoring Script

```python
import asyncio
import time
from Infrastructure.vector_stores.faiss_store import FAISSVectorStore

async def monitor_vector_store():
    vector_store = FAISSVectorStore()

    print("🔄 Real-time Vector Store Monitor")
    print("Press Ctrl+C to stop")
    print("=" * 50)

    try:
        while True:
            # Check status
            is_init = vector_store.is_initialized()
            doc_count = await vector_store.get_document_count() if is_init else 0

            # Test search
            if is_init and doc_count > 0:
                results = await vector_store.search_similar(
                    "test query", k=1, similarity_threshold=0.1
                )
                search_status = f"✅ {len(results)} results"
            else:
                search_status = "❌ No search possible"

            print(f"[{time.strftime('%H:%M:%S')}] "
                  f"Init: {is_init} | Docs: {doc_count} | Search: {search_status}")

            await asyncio.sleep(5)

    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")

if __name__ == "__main__":
    asyncio.run(monitor_vector_store())
```

---

## 🛡️ Prevention Strategies

### 1. Robust Upload Verification

**Modify document controller** to verify processing:
```python
# In DocumentController
async def upload_document_with_verification(self, file: UploadFile):
    # Upload
    result = await self.upload_document(file)

    # Wait and verify
    max_wait = 60  # seconds
    for _ in range(max_wait):
        await asyncio.sleep(1)
        doc_count = await self.vector_store.get_document_count()
        if doc_count > 0:
            break

    return {
        **result,
        "verification": {
            "documents_indexed": doc_count,
            "processing_complete": doc_count > 0
        }
    }
```

### 2. Adaptive Similarity Thresholds

**Improve search with fallback thresholds:**
```python
# In MCQGenerationService
async def search_with_fallback(self, query: str, k: int = 5):
    """Search with progressively lower thresholds"""
    thresholds = [0.3, 0.2, 0.1, 0.05]

    for threshold in thresholds:
        results = await self.vector_store_repository.search_similar(
            query, k=k, similarity_threshold=threshold
        )
        if results:
            print(f"Found {len(results)} results with threshold {threshold}")
            return results

    # Last resort: get any available documents
    results = await self.vector_store_repository.search_similar(
        query, k=k, similarity_threshold=0.01
    )

    if not results:
        raise ValueError(f"No relevant context found for topic: {query}")

    return results
```

### 3. Enhanced Error Messages

**Provide detailed error context:**
```python
# In MCQGenerationService
async def generate_mcq(self, request: MCQRequestDTO):
    try:
        # Get context
        context_docs = await self.search_with_fallback(request.topic)

    except ValueError as e:
        # Provide debugging info
        doc_count = await self.vector_store_repository.get_document_count()
        is_init = self.vector_store_repository.is_initialized()

        detailed_error = (
            f"MCQ generation failed for topic '{request.topic}'. "
            f"Debug info: Vector store initialized: {is_init}, "
            f"Documents available: {doc_count}. "
            f"Suggestions: 1) Upload relevant documents, "
            f"2) Use more general topics, "
            f"3) Check if documents finished processing."
        )
        raise ValueError(detailed_error)
```

### 4. Health Check Enhancements

**Add vector store health details:**
```python
# In SystemController
async def get_detailed_health(self):
    health = await self.get_health_check()

    # Add vector store details
    if self.vector_store:
        vector_health = {
            "document_count": await self.vector_store.get_document_count(),
            "embedding_model": self.vector_store.embedding_model_name,
            "last_search_test": await self._test_search()
        }
        health["vector_store_details"] = vector_health

    return health

async def _test_search(self):
    """Test if search is working"""
    try:
        results = await self.vector_store.search_similar(
            "test", k=1, similarity_threshold=0.1
        )
        return {
            "status": "working",
            "test_results": len(results)
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }
```

---

## 🎓 Learning Intuition

### Why This Error Happens - The Big Picture

Think of the vector store like a **library catalog system**:

1. **Books (Documents)** need to be **cataloged (embedded)** before you can find them
2. **Your search query** needs to **match the catalog entries** (similar embeddings)
3. **The librarian (similarity threshold)** decides what's "close enough" to your request

### Common Student Mistakes

#### 1. **Impatience** ⏰
```
Student: "I uploaded the file 2 seconds ago, why doesn't search work?"
Reality: Background processing takes 30-60 seconds
```

#### 2. **Wrong Expectations** 🎯
```
Student: "I searched for 'Quantum Physics' in a spam detection PDF"
Reality: Search only finds what actually exists in your documents
```

#### 3. **Ignoring Basics** 🔍
```
Student: "The complex search algorithm isn't working!"
Reality: No documents were uploaded in the first place
```

### Building Debugging Intuition

**Ask yourself these questions in order:**

1. **"Is there any data?"** → Check document count
2. **"Is the data processed?"** → Wait for background processing
3. **"Am I searching for the right thing?"** → Use content from your PDF
4. **"Are my parameters too strict?"** → Lower similarity threshold
5. **"Is the system working at all?"** → Check health endpoints

### Mental Model for Vector Search

```
Your Query: "Machine Learning"
    ↓ (Embedding Model)
Vector: [0.1, -0.3, 0.7, ...]
    ↓ (Similarity Search)
Compare with all document vectors
    ↓ (Distance Calculation)
Find closest matches
    ↓ (Threshold Filter)
Keep only good enough matches
    ↓ (Return Results)
Context for MCQ generation
```

**If any step fails → No results!**

---

## 🚀 Quick Fix Checklist

When you get "No relevant context found":

- [ ] **Check health**: `curl http://localhost:8000/health`
- [ ] **Check documents**: `curl http://localhost:8000/api/v1/documents/stats`
- [ ] **Wait 30 seconds** after upload
- [ ] **Use specific queries** from your PDF content
- [ ] **Lower similarity threshold** in testing
- [ ] **Check logs** for processing errors
- [ ] **Restart system** if needed

### Emergency Debug Commands

```bash
# Quick health check
curl -s http://localhost:8000/health | jq .status

# Quick document check
curl -s http://localhost:8000/api/v1/documents/stats | jq .document_count

# Quick upload test
curl -X POST http://localhost:8000/api/v1/documents/upload -F "file=@pdfs/Project_2_2_SpamClassifier_AIOTimeSeries.pdf"

# Wait and test
sleep 30 && curl -X POST http://localhost:8000/api/v1/mcq/generate -H "Content-Type: application/json" -d '{"topic":"Spam","question_type":"single_choice"}'
```

---

**Remember**: Debugging is like detective work. Start with the obvious, gather evidence systematically, and don't jump to complex solutions until you've ruled out simple causes. The vector store retrieval system has a clear data flow - trace it step by step! 🕵️‍♂️

**Happy Debugging!** 🐛➡️✅
