from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss                   
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATASET_PATH = Path("text2sql_epi_dataset_omop.xlsx")   
ARTIFACT_DIR = Path("rag_index")                        
INDEX_FILE   = ARTIFACT_DIR / "faiss.index"
META_FILE    = ARTIFACT_DIR / "metadata.pkl"


EMBED_MODEL_NAME = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"  
K_NEIGHBOURS     = 5                                         

class MedicalSQLRetriever:
    

    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.index: faiss.Index | None = None
        self.metadata: List[dict] | None = None  # one dict per vector

    def build(self, dataset_path: Path, question_cols: list[str] | None = None) -> None:
        print("[•] Loading dataset…")
        df = pd.read_excel(dataset_path)


        if question_cols is None:
            question_cols = [c for c in df.columns if c.upper().startswith("QUESTION")]
            if not question_cols:
                raise ValueError("No QUESTION* columns found; pass --question-cols.")
        else:
            missing = [c for c in question_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Columns not found: {missing}")

        sql_col_candidates = [c for c in df.columns if "QUERY" in c.upper() and "RUNNABLE" in c.upper()]
        if not sql_col_candidates:
            raise ValueError("Couldn't spot a *_RUNNABLE SQL column.")
        sql_col = sql_col_candidates[0]


        questions, metadatas = [], []
        for _, row in df.iterrows():
            variants = [str(row[c]) for c in question_cols if pd.notna(row[c])]
            for variant in variants:
                questions.append(variant)
                metadatas.append(
                    {
                        "row_id": int(row["ID"]) if "ID" in row else None,
                        "canonical_question": str(row[question_cols[0]]),
                        "sql": str(row[sql_col]),
                    }
                )

        print(f"[•] Total question variants: {len(questions):,}")
        print("[•] Computing embeddings…")
        embeds = self.model.encode(questions, show_progress_bar=True, convert_to_numpy=True)
        dim = embeds.shape[1]
        faiss.normalize_L2(embeds)

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeds)
        self.metadata = metadatas


        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(INDEX_FILE))
        with META_FILE.open("wb") as fp:
            pickle.dump(self.metadata, fp)
        print("[✔] Index built – vectors:", self.index.ntotal)


    def load(self) -> None:
        if not INDEX_FILE.exists():
            raise FileNotFoundError("Index not found; run with --build first.")
        self.index = faiss.read_index(str(INDEX_FILE))
        with META_FILE.open("rb") as fp:
            self.metadata = pickle.load(fp)


    def query(self, text: str, k: int = K_NEIGHBOURS) -> List[Tuple[float, dict]]:
        assert self.index is not None and self.metadata is not None, "Call .load() or .build() first."
        q_emb = self.model.encode([text], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        scores, idxs = self.index.search(q_emb, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((float(score), self.metadata[idx]))
        return results


def _cli() -> None:
    p = argparse.ArgumentParser(description="Semantic medical‑SQL retriever")
    p.add_argument("question", help="Free‑text question to search for (any language)")
    p.add_argument("--build", action="store_true", help="(Re)build the FAISS index first")
    p.add_argument("--topk", type=int, default=K_NEIGHBOURS, help="# neighbours to return")
    p.add_argument("--question-cols", default=None, help="Comma‑separated list of question columns")
    
    args = p.parse_args()

    qcols = args.question_cols.split(",") if args.question_cols else None

    retriever = MedicalSQLRetriever()
    if args.build or not INDEX_FILE.exists():
        retriever.build(DATASET_PATH, question_cols=qcols)
    else:
        retriever.load()

    print("\n[•] Query:", args.question)
    results = retriever.query(args.question, k=args.topk)
    if not results:
        print("No similar questions found")
        return

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for rank, (score, meta) in enumerate(results, start=1):
        print(f"#{rank} – score {score:.3f}")
        print("Q:", meta["canonical_question"])
        snippet = meta["sql"][:200] + ("…" if len(meta["sql"]) > 200 else "")
        print("SQL (truncated):", snippet)
        print("―" * 55)
        if rank == 1:
            print("\n>>> Full SQL for best match:\n")
            print(meta["sql"])
            print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    _cli()
