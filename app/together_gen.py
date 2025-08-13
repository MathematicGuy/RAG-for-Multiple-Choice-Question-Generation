import re
import random
import numpy as np
import os
from typing import List, Tuple, Dict, Any
import pdfplumber
from sentence_transformers import SentenceTransformer
from utils import generate_mcqs_from_text, _post_chat, _safe_extract_json

try:
    import faiss
    _HAS_FAISS = True
except:
    _HAS_FAISS = False

class RAGMCQ:
    def __init__(
        self,
        embedder_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        hf_model: str = "openai/gpt-oss-20b",
    ):
        self.embedder = SentenceTransformer(embedder_model)
        self.hf_model = hf_model
        self.embeddings = None   # np.array of shape (N, D)
        self.texts = []          # list of chunk texts
        self.metadata = []       # list of dicts (page, chunk_id, char_range)
        self.index = None
        self.dim = self.embedder.get_sentence_embedding_dimension()

    def extract_pages(self, pdf_path: str) -> List[str]:
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                txt = p.extract_text() or ""
                pages.append(txt.strip())
        return pages

    def chunk_text(self, text: str, max_chars: int = 1200) -> List[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= max_chars:
            return [text]

        # split by sentence-like boundaries
        sentences = re.split(r'(?<=[\.\?\!])\s+', text)
        chunks = []
        cur = ""
        for s in sentences:
            if len(cur) + len(s) + 1 <= max_chars:
                cur += (" " if cur else "") + s
            else:
                if cur:
                    chunks.append(cur)
                cur = s
        if cur:
            chunks.append(cur)

        # if still too long, hard-split
        final = []
        for c in chunks:
            if len(c) <= max_chars:
                final.append(c)
            else:
                for i in range(0, len(c), max_chars):
                    final.append(c[i:i+max_chars])
        return final

    def build_index_from_pdf(self, pdf_path: str, max_chars: int = 1200):
        pages = self.extract_pages(pdf_path)
        self.texts = []
        self.metadata = []

        for p_idx, page_text in enumerate(pages, start=1):
            chunks = self.chunk_text(page_text or "", max_chars=max_chars)
            for cid, ch in enumerate(chunks, start=1):
                self.texts.append(ch)
                self.metadata.append({"page": p_idx, "chunk_id": cid, "length": len(ch)})

        if not self.texts:
            raise RuntimeError("No text extracted from PDF.")

        # compute embeddings
        emb = self.embedder.encode(self.texts, convert_to_numpy=True, show_progress_bar=True)
        self.embeddings = emb.astype("float32")
        self._build_faiss_index()

    def _build_faiss_index(self):
        if _HAS_FAISS:
            d = self.embeddings.shape[1]
            index = faiss.IndexFlatIP(d)  # inner product -> cosine if vectors normalized
            faiss.normalize_L2(self.embeddings)
            index.add(self.embeddings)
            self.index = index
        else:
            # store normalized embeddings and use brute-force numpy
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10
            self.embeddings = self.embeddings / norms
            self.index = None

    def _retrieve(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        q_emb = self.embedder.encode([query], convert_to_numpy=True).astype("float32")

        if _HAS_FAISS:
            faiss.normalize_L2(q_emb)
            D, I = self.index.search(q_emb, top_k)
            # D are inner products; return list of (idx, score)
            return [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i != -1]
        else:
            qn = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)
            sims = (self.embeddings @ qn.T).squeeze(axis=1)
            idxs = np.argsort(-sims)[:top_k]
            return [(int(i), float(sims[i])) for i in idxs]

    def generate_from_pdf(
        self,
        pdf_path: str,
        n_questions: int = 3,
        mode: str = "rag", # per_page or rag
        questions_per_page: int = 1, # for per_page mode
        top_k: int = 3, # chunks to retrieve for each question in rag mode
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        # build index
        self.build_index_from_pdf(pdf_path)

        output: Dict[str, Any] = {}
        qcount = 0

        if mode == "per_page":
            # iterate pages -> chunks
            for idx, meta in enumerate(self.metadata):
                chunk_text = self.texts[idx]

                if not chunk_text.strip():
                    continue
                to_gen = questions_per_page

                # ask generator
                try:
                    mcq_block = generate_mcqs_from_text(
                        chunk_text, n=to_gen, model=self.hf_model, temperature=temperature
                    )
                except Exception as e:
                    # skip this chunk if generator fails
                    print(f"Generator failed on page {meta['page']} chunk {meta['chunk_id']}: {e}")
                    continue

                for item in sorted(mcq_block.keys(), key=lambda x: int(x)):
                    qcount += 1
                    output[str(qcount)] = mcq_block[item]
                    if qcount >= n_questions:
                        return output

            return output

        elif mode == "rag":
            # strategy: create a few natural short queries by sampling sentences or using chunk summaries.
            # create queries by sampling chunk text sentences.
            # stop when n_questions reached or max_attempts exceeded.
            attempts = 0
            max_attempts = n_questions * 4

            while qcount < n_questions and attempts < max_attempts:
                attempts += 1
                # create a seed query: pick a random chunk, pick a sentence from it
                seed_idx = random.randrange(len(self.texts))
                chunk = self.texts[seed_idx]
                sents = re.split(r'(?<=[\.\?\!])\s+', chunk)
                seed_sent = random.choice([s for s in sents if len(s.strip()) > 20]) if sents else chunk[:200]
                query = f"Create questions about: {seed_sent}"

                # retrieve top_k chunks
                retrieved = self._retrieve(query, top_k=top_k)
                context_parts = []
                for ridx, score in retrieved:
                    md = self.metadata[ridx]
                    context_parts.append(f"[page {md['page']}] {self.texts[ridx]}")
                context = "\n\n".join(context_parts)

                # call generator for 1 question (or small batch) with the retrieved context
                try:
                    # request 1 question at a time to keep diversity
                    mcq_block = generate_mcqs_from_text(
                        context, n=1, model=self.hf_model, temperature=temperature
                    )
                except Exception as e:
                    print(f"Generator failed during RAG attempt {attempts}: {e}")
                    continue

                # append result(s)
                for item in sorted(mcq_block.keys(), key=lambda x: int(x)):
                    qcount += 1
                    output[str(qcount)] = mcq_block[item]
                    if qcount >= n_questions:
                        return output

            return output
        else:
            raise ValueError("mode must be 'per_page' or 'rag'.")

    def validate_mcqs(
        self,
        mcqs: Dict[str, Any],
        top_k: int = 4,
        similarity_threshold: float = 0.7,
        evidence_score_cutoff: float = 0.45,
        use_model_verification: bool = True,
        model_verification_temperature: float = 0.0,
    ) -> Dict[str, Any]:
        if self.embeddings is None or not self.texts:
            raise RuntimeError("Index/embeddings not built. Run build_index_from_pdf() first.")

        # ensure embeddings are normalized locally for cosine similarity (do not modify original unexpectedly)
        emb = self.embeddings.astype("float32")
        emb_norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10
        emb_normalized = emb / emb_norms  # shape (N, D)

        report: Dict[str, Any] = {}

        # helper: semantic similarity search on statement -> returns list of (idx, score)
        def semantic_search(statement: str, k: int = top_k):
            q_emb = self.embedder.encode([statement], convert_to_numpy=True).astype("float32")
            q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)
            sims = (emb_normalized @ q_emb.T).squeeze()  # shape (N,)
            top_idx = np.argsort(-sims)[:k]
            return [(int(i), float(sims[i])) for i in top_idx]

        # helper: verify with model (strict JSON in response)
        def _verify_with_model(question_text: str, options: Dict[str, str], correct_text: str, context_text: str):
            system = {
                "role": "system",
                "content": (
                    "Bạn là một trợ lý đánh giá tính thực chứng của câu hỏi trắc nghiệm dựa trên đoạn văn được cung cấp. "
                    "Hãy trả lời DUY NHẤT bằng JSON hợp lệ (không có văn bản khác) theo schema:\n\n"
                    "{\n"
                    '  "supported": true/false,            # câu trả lời đúng có được nội dung chứng thực không\n'
                    '  "confidence": 0.0-1.0,              # mức độ tự tin (số)\n'
                    '  "evidence": "cụm văn bản ngắn làm bằng chứng hoặc trích dẫn",\n'
                    '  "reason": "ngắn gọn, vì sao supported hoặc không"\n'
                    "}\n\n"
                    "Luôn dựa chỉ trên nội dung trong trường 'Context' dưới đây. Nếu nội dung không chứa bằng chứng, trả supported: false."
                )
            }
            user = {
                "role": "user",
                "content": (
                    "Question:\n" + question_text + "\n\n"
                    "Options:\n" + "\n".join([f"{k}: {v}" for k, v in options.items()]) + "\n\n"
                    "Claimed correct answer (full text):\n" + correct_text + "\n\n"
                    "Context:\n" + context_text + "\n\n"
                    "Hãy trả lời JSON như yêu cầu."
                )
            }

            raw = _post_chat([system, user], model=self.hf_model, temperature=model_verification_temperature)

            # parse JSON object in response
            try:
                parsed = _safe_extract_json(raw)
            except Exception as e:
                return {"error": f"Model verification failed to return JSON: {e}", "raw": raw}
            return parsed

        # iterate MCQs
        for qid, item in mcqs.items():
            q_text = item.get("mcq", "").strip()
            options = item.get("options", {})
            correct_text = item.get("correct", "").strip()

            # form a short declarative statement to embed: "Question: ... Answer: <correct>"
            statement = f"{q_text} Answer: {correct_text}"

            retrieved = semantic_search(statement, k=top_k)
            evidence_list = []
            max_sim = 0.0
            for idx, score in retrieved:
                if score >= evidence_score_cutoff:
                    evidence_list.append({
                        "idx": idx,
                        "page": self.metadata[idx].get("page", None),
                        "score": float(score),
                        "text": (self.texts[idx][:1000] + ("..." if len(self.texts[idx]) > 1000 else "")),
                    })

                if score > max_sim:
                    max_sim = float(score)

            supported_by_embeddings = max_sim >= similarity_threshold
            model_verdict = None

            # if embeddings suggest not supported and user wants model verification, call model
            if not supported_by_embeddings and use_model_verification:
                # build a context string from top retrieved chunks (regardless of cutoff)
                context_parts = []
                for ridx, sc in retrieved:
                    md = self.metadata[ridx]
                    context_parts.append(f"[page {md.get('page')}] {self.texts[ridx]}")
                context_text = "\n\n".join(context_parts)

                try:
                    parsed = _verify_with_model(q_text, options, correct_text, context_text)
                    model_verdict = parsed
                except Exception as e:
                    model_verdict = {"error": f"verification exception: {e}"}

            report[qid] = {
                "supported_by_embeddings": bool(supported_by_embeddings),
                "max_similarity": float(max_sim),
                "evidence": evidence_list,
                "model_verdict": model_verdict,
            }

        return report
