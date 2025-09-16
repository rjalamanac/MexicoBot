#!/usr/bin/env python3
"""
Uso:
    pip install -r requirements.txt
    python contract_bot.py --build --pdf_folder ./pdfs
    python contract_bot.py --ask

Opciones:
    --build         : procesa PDFs y construye índices
    --ask           : entra en modo interactivo para hacer preguntas
    --rebuild       : forzar reconstrucción del índice aunque exista
    --datafile PATH : ruta para guardar datos (default: ./contract_store.pkl)

"""

import os
import sys
import argparse
import re
import pickle
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm

# OCR / PDF
try:
    from pdf2image import convert_from_path
except Exception as e:
    convert_from_path = None
from PIL import Image
import pytesseract

# NLP / Embeddings / QA / FAISS / BM25
import numpy as np
from sentence_transformers import SentenceTransformer, util
import faiss
from transformers import pipeline
from rank_bm25 import BM25Okapi

# -----------------------
# Config
# -----------------------
PDF2IMAGE_DPI = 300
CHUNK_WORD_SIZE = 200      # tamaño base de chunk; luego intentamos split semántico
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
QA_MODEL_NAME = "mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
DATA_STORE = "contract_store.pkl"

# -----------------------
# Utilities: OCR + PDF -> text
# -----------------------
def pdf_to_images(pdf_path: str, dpi:int = PDF2IMAGE_DPI) -> List[Image.Image]:
    if convert_from_path is None:
        raise RuntimeError("pdf2image not available. Install pdf2image and poppler-utils.")
    images = convert_from_path(pdf_path, dpi=dpi)
    return images

def ocr_image_to_text(image: Image.Image, lang="spa") -> str:
    # pytesseract expects PIL Image
    text = pytesseract.image_to_string(image, lang=lang)
    return text

def pdf_to_text(pdf_path: str, lang="spa") -> str:
    images = pdf_to_images(pdf_path)
    texts = []
    for img in images:
        try:
            t = ocr_image_to_text(img, lang=lang)
        except Exception as e:
            print(f"[WARN] pytesseract failed on a page: {e}")
            t = ""
        texts.append(t)
    return "\n".join(texts)

# -----------------------
# Preprocessing: chunking & semantic section splits
# -----------------------
def split_by_headings(text: str) -> List[str]:
    """
    Intenta separar por títulos comunes en contratos (Ej.: 'Artículo', 'Cláusula', 'CONTRATO', line breaks grandes)
    Fallback: chunk por tamaño de palabras.
    """
    # Normalizar saltos de línea
    text = re.sub(r'\r\n', '\n', text)
    # Split using common section keywords
    headings = re.split(r'(\n\s*(Artículo|ARTICULO|Cláusula|CLÁUSULA|CONTRATO|Contrato|ART\.)[^\\n]*)', text)
    # headings will include separators; reconstruct sensible blocks
    blocks = []
    if len(headings) > 1:
        # Combine pairs (separator + text) heuristically
        curr = ""
        for part in headings:
            if re.match(r'\n\s*(Artículo|ARTICULO|Cláusula|CLÁUSULA|CONTRATO|Contrato|ART\.)', part or ""):
                # treat as new header
                if curr.strip():
                    blocks.append(curr.strip())
                curr = part
            else:
                curr += part
        if curr.strip():
            blocks.append(curr.strip())
    else:
        # fallback: chunk by word count
        words = text.split()
        blocks = [" ".join(words[i:i+CHUNK_WORD_SIZE]) for i in range(0, len(words), CHUNK_WORD_SIZE)]
    # ensure no too-large blocks: further chunk any block > 2*CHUNK_WORD_SIZE
    final = []
    for b in blocks:
        w = b.split()
        if len(w) > 2*CHUNK_WORD_SIZE:
            for i in range(0, len(w), CHUNK_WORD_SIZE):
                final.append(" ".join(w[i:i+CHUNK_WORD_SIZE]))
        else:
            final.append(b)
    return final

# -----------------------
# Entity extraction (heuristic / regex)
# -----------------------
DATE_RE = re.compile(r'(\d{1,2}\s+de\s+[A-Za-záéíóúÁÉÍÓÚñÑ]+\s+\d{4})', re.IGNORECASE)
CONTRACT_RE = re.compile(r'(?:Contrato|CONTRATO|contrato)\s*[:#]?\s*(\d{2,20})', re.IGNORECASE)
SINDICATE_RE = re.compile(r'(Sindicat[oa][\s:–-]*[A-Za-z0-9ÁÉÍÓÚÜñÑ\-\s]{2,60})', re.IGNORECASE)

def extract_entities(text: str) -> Dict[str, Any]:
    ents = {}
    m = CONTRACT_RE.search(text)
    if m:
        ents['contract_number'] = m.group(1).strip()
    d = DATE_RE.search(text)
    if d:
        ents['last_date_mentioned'] = d.group(1).strip()
    s = SINDICATE_RE.search(text)
    if s:
        ents['union'] = s.group(1).strip()
    return ents

# -----------------------
# Index building
# -----------------------
class ContractStore:
    def __init__(self):
        self.chunks: List[str] = []         # chunk texts
        self.metadata: List[Dict] = []      # meta per chunk: {doc_name, contract_number, union, date, chunk_id}
        self.bm25 = None                    # BM25 index (built from tokenized chunks)
        self.embed_model = None
        self.embeddings = None              # numpy array
        self.faiss_index = None
        self.faiss_dim = None
        self.qa_pipeline = None

    def build_from_folder(self, folder: str, lang="spa"):
        folder = Path(folder)
        assert folder.exists(), f"PDF folder {folder} does not exist"
        pdf_files = list(folder.glob("*.pdf"))
        if not pdf_files:
            print("[WARN] No PDFs found in folder.")
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                text = pdf_to_text(str(pdf_path), lang=lang)
            except Exception as e:
                print(f"[ERROR] Failed converting PDF {pdf_path}: {e}")
                continue
            blocks = split_by_headings(text)
            doc_entities = extract_entities(text)
            for i, block in enumerate(blocks):
                chunk_id = f"{pdf_path.name}__{i}"
                meta = {
                    "doc_name": pdf_path.name,
                    "chunk_id": chunk_id,
                    "contract_number": doc_entities.get("contract_number"),
                    "union": doc_entities.get("union"),
                    "last_date": doc_entities.get("last_date_mentioned")
                }
                self.chunks.append(block)
                self.metadata.append(meta)
        # build BM25
        tokenized = [chunk.lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized)
        print(f"[INFO] Built BM25 over {len(self.chunks)} chunks")

    def build_embeddings(self):
        if self.embed_model is None:
            print("[INFO] Loading embedding model...")
            self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        print("[INFO] Encoding chunks to embeddings...")
        embs = self.embed_model.encode(self.chunks, show_progress_bar=True, convert_to_numpy=True)
        self.embeddings = np.array(embs).astype("float32")
        self.faiss_dim = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(self.faiss_dim)
        self.faiss_index.add(self.embeddings)
        print(f"[INFO] FAISS index built. dim={self.faiss_dim}, n={self.embeddings.shape[0]}")

    def load_qa_pipeline(self):
        if self.qa_pipeline is None:
            print("[INFO] Loading QA pipeline (Spanish)...")
            self.qa_pipeline = pipeline("question-answering", model=QA_MODEL_NAME)

    def save(self, path: str):
        # save metadata and chunks and BM25 internal data; do not pickle models
        obj = {
            "chunks": self.chunks,
            "metadata": self.metadata
        }
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        # Save FAISS index and embeddings
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, path + ".faiss")
            np.save(path + ".embeddings.npy", self.embeddings)
        print(f"[INFO] Store saved to {path} (and .faiss/.embeddings.npy)")

    def load(self, path: str):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.chunks = obj["chunks"]
        self.metadata = obj["metadata"]
        # rebuild BM25
        tokenized = [chunk.lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized)
        # load embeddings & faiss if present
        if os.path.exists(path + ".embeddings.npy") and os.path.exists(path + ".faiss"):
            self.embeddings = np.load(path + ".embeddings.npy")
            self.faiss_index = faiss.read_index(path + ".faiss")
            self.faiss_dim = self.embeddings.shape[1]
            self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)  # needed for queries
            print(f"[INFO] Loaded embeddings/FAISS index with n={self.embeddings.shape[0]}")
        else:
            print("[WARN] No embeddings/FAISS found; run build_embeddings() to create them")

# -----------------------
# Querying logic (hybrid)
# -----------------------
def hybrid_retrieve(store: ContractStore, question: str, top_k_bm25=5, top_k_faiss=5) -> List[Dict]:
    """
    Returns a list of candidate {chunk, meta, score} ordered by relevance.
    Steps:
       1) BM25 top_k_bm25 (fast keyword retrieval)
       2) FAISS: encode question and search top_k_faiss by embedding
       3) merge candidates and compute a combined score (bm25_score normalized + embedding similarity)
    """
    q = question.lower().split()
    bm25_scores = store.bm25.get_scores(q)
    bm25_top_idx = np.argsort(bm25_scores)[::-1][:top_k_bm25]
    bm25_candidates = {int(i): float(bm25_scores[int(i)]) for i in bm25_top_idx if bm25_scores[int(i)]>0}

    # faiss
    embed_model = store.embed_model or SentenceTransformer(EMBED_MODEL_NAME)
    q_emb = embed_model.encode([question], convert_to_numpy=True)
    faiss_candidates = {}
    if store.faiss_index is not None:
        D, I = store.faiss_index.search(np.array(q_emb).astype("float32"), top_k_faiss)
        # D are L2 distances; convert to similarity by negative distance
        for dist, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            faiss_candidates[int(idx)] = float(-dist)

    # Merge indices
    candidate_idxs = set(list(bm25_candidates.keys()) + list(faiss_candidates.keys()))
    results = []
    # normalize bm25 scores
    bm_vals = np.array(list(bm25_candidates.values())) if bm25_candidates else np.array([0.0])
    bm_min, bm_max = float(bm_vals.min()) if len(bm_vals)>0 else 0.0, float(bm_vals.max()) if len(bm_vals)>0 else 1.0
    for idx in candidate_idxs:
        bm = bm25_candidates.get(idx, 0.0)
        if bm_max>bm_min:
            bm_norm = (bm - bm_min)/(bm_max - bm_min)
        else:
            bm_norm = 0.0
        fa = faiss_candidates.get(idx, 0.0)
        score = 0.6 * bm_norm + 0.4 * fa  # weighted combination
        results.append({
            "idx": idx,
            "chunk": store.chunks[idx],
            "meta": store.metadata[idx],
            "score": score,
            "bm25": bm,
            "faiss": fa
        })
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results

def answer_question(store: ContractStore, question: str, top_k_contexts=3) -> Dict:
    # Heuristic: if question asks "número" or "número de contrato", try to return metadata directly
    qlow = question.lower()
    if "número" in qlow or "numero" in qlow or "nº" in qlow:
        # Try to find any contract_number in metadata
        for meta in store.metadata:
            if meta.get("contract_number"):
                return {"answer": meta["contract_number"], "source": meta}
    # Hybrid retrieve
    cand = hybrid_retrieve(store, question, top_k_bm25=10, top_k_faiss=10)
    if not cand:
        return {"answer": "No encontré información relevante.", "source": None}
    # Combine top contexts
    contexts = []
    used = set()
    for item in cand[:top_k_contexts]:
        if item["idx"] not in used:
            contexts.append(item["chunk"])
            used.add(item["idx"])
    combined_context = "\n\n".join(contexts)
    # QA pipeline
    store.load_qa_pipeline()
    try:
        qa_res = store.qa_pipeline(question=question, context=combined_context)
        ans = qa_res.get("answer", "").strip()
        score = qa_res.get("score", 0.0)
    except Exception as e:
        print(f"[WARN] QA model failed: {e}")
        ans = ""
        score = 0.0
    # If QA returns nothing useful, fallback to top metadata fields
    if (not ans) or (len(ans) < 2):
        # try to pick union or date from top candidates
        for item in cand:
            m = item["meta"]
            if m.get("contract_number"):
                return {"answer": m["contract_number"], "source": m}
            if m.get("union") and ("sindicat" in qlow or "sindicato" in qlow or "firma" in qlow):
                return {"answer": m["union"], "source": m}
        return {"answer": "No tengo una respuesta clara, revisa los documentos: " + ", ".join([c['meta']['doc_name'] for c in cand[:3]]), "source": cand[:3]}
    return {"answer": ans, "score": score, "source": cand[:3]}

# -----------------------
# CLI / Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Process PDFs and build store")
    parser.add_argument("--pdf_folder", type=str, default="./pdfs", help="Folder with PDFs")
    parser.add_argument("--ask", action="store_true", help="Interactive ask mode")
    parser.add_argument("--datafile", type=str, default=DATA_STORE, help="Path to save/store data")
    parser.add_argument("--rebuild", action="store_true", help="Forzar reconstrucción del índice")
    args = parser.parse_args()

    store = ContractStore()
    datafile = args.datafile

    if args.build:
        print("[STEP] Building store from PDFs...")
        store.build_from_folder(args.pdf_folder)
        store.build_embeddings()
        store.save(datafile)
        print("[DONE] Build complete.")
        return

    # If not building, try to load existing store
    if os.path.exists(datafile):
        print("[INFO] Loading existing store...")
        store.load(datafile)
        # if embeddings not present, build them
        if store.faiss_index is None:
            store.build_embeddings()
            store.save(datafile)
    else:
        print(f"[ERROR] No store found at {datafile}. Run with --build --pdf_folder ./pdfs first.")
        return

    if args.ask:
        print("Modo interactivo. Escribe 'salir' para terminar.")
        while True:
            q = input("\nPregunta: ").strip()
            if q.lower() in ("salir", "exit", "quit"):
                break
            res = answer_question(store, q)
            print("\nRespuesta:")
            print(res.get("answer"))
            # Optional: show sources
            src = res.get("source")
            if src:
                print("\nFuentes:")
                if isinstance(src, list):
                    for s in src:
                        meta = s.get("meta", s)
                        print(f"- {meta.get('doc_name')} (chunk {s.get('idx')})")
                elif isinstance(src, dict):
                    print(f"- {src.get('doc_name')}")
        print("Bye.")
    else:
        print("Carga completada. Ejecuta con --ask para hacer preguntas o --build para reconstruir índices.")

if __name__ == "__main__":
    main()
