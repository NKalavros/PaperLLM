#!/usr/bin/env python3
"""
Semantic similarity analysis for questions and answers.

Inputs:
- requests_questions.log (JSONL)
- requests_answers.log (JSONL)
- Optional Talk transcripts: TalkN.txt (for labeling & optional centroids)

Outputs (under analysis_plots/):
- Heatmaps, histograms, PCA (SVD) scatter plots colored by talk/asker
- Dendrograms (hierarchical clustering via SciPy)

Notes:
- Embeddings via OpenAI (env OPENAI_API_KEY*). Uses simple local JSON cache to avoid re-embedding.
- No scikit-learn required; PCA implemented with NumPy SVD, clustering with SciPy.
"""

from __future__ import annotations

import os
import json
import time
import math
import hashlib
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    def load_dotenv(*args, **kwargs):
        return False


# ----------------------------- Config -----------------------------

EMBED_MODEL_DEFAULT = os.environ.get("EMBED_MODEL", "text-embedding-3-large")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "text_cache")
CACHE_PATH = os.path.join(CACHE_DIR, "embeddings_cache.jsonl")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "analysis_plots")

QUESTIONS_LOG = os.path.join(os.path.dirname(__file__), "requests_questions.log")
ANSWERS_LOG = os.path.join(os.path.dirname(__file__), "requests_answers.log")


# --------------------------- Utilities ----------------------------

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def pairwise_cosine_matrix(X: np.ndarray) -> np.ndarray:
    # Normalize rows
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    return Xn @ Xn.T


def pca_project(X: np.ndarray, k: int = 2) -> np.ndarray:
    # Center
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    # SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :k] * S[:k]


# ------------------------- Embedding Cache ------------------------

class EmbeddingCache:
    def __init__(self, cache_path: str):
        self.path = cache_path
        self._mem: Dict[Tuple[str, str], List[float]] = {}
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    for line in f:
                        rec = json.loads(line)
                        self._mem[(rec["hash"], rec["model"])]= rec["embedding"]
            except Exception:
                pass

    def get(self, text: str, model: str) -> Optional[List[float]]:
        key = (sha256_text(text), model)
        return self._mem.get(key)

    def put(self, text: str, model: str, embedding: List[float]) -> None:
        ensure_dir(os.path.dirname(self.path))
        rec = {
            "hash": sha256_text(text),
            "model": model,
            "embedding": embedding,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        self._mem[(rec["hash"], model)] = embedding


def batched(iterable, n):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


# --------------------------- OpenAI API ---------------------------

class OpenAIEmbedder:
    def __init__(self, model: str):
        # Try multiple keys OPENAI_API_KEY, OPENAI_API_KEY2, ... like app.py
        from openai import OpenAI  # type: ignore

        keys = [os.getenv(f"OPENAI_API_KEY{i}") for i in range(1, 6)]
        keys = [k for k in keys if k]
        if not keys:
            # Fall back to default env var
            k = os.getenv("OPENAI_API_KEY")
            if k:
                keys = [k]

        if not keys:
            raise RuntimeError("No OpenAI API keys found in environment.")

        self.clients = [OpenAI(api_key=k) for k in keys]
        self.model = model
        self._client_idx = 0

    def _next_client(self):
        c = self.clients[self._client_idx]
        self._client_idx = (self._client_idx + 1) % len(self.clients)
        return c

    def embed_texts(self, texts: List[str], batch_size: int = 128, sleep_on_rate_limit: float = 2.0) -> List[List[float]]:
        out: List[List[float]] = []
        for batch in batched(texts, batch_size):
            for attempt in range(5):
                client = self._next_client()
                try:
                    resp = client.embeddings.create(model=self.model, input=batch)
                    out.extend([d.embedding for d in resp.data])
                    break
                except Exception as e:
                    # Simple backoff on rate limits/network issues
                    if attempt == 4:
                        raise
                    time.sleep(sleep_on_rate_limit * (attempt + 1))
        return out


# ------------------------------ Data -----------------------------

def load_questions(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            rid = rec.get("request_id")
            prompt = rec.get("prompt") or ""
            if not rid or not prompt:
                continue
            # Filter filler prompts
            if prompt.strip().lower().startswith("type in your question here"):
                continue
            rows.append({
                "request_id": rid,
                "question": prompt,
                "nickname": rec.get("nickname"),
                "asker_nickname": rec.get("nickname"),  # alias for joins
                "difficulty": rec.get("question_difficulty"),
                "file": rec.get("file"),
                "talk": rec.get("speaker"),
                "text_preview": rec.get("text_preview"),
            })
    df = pd.DataFrame(rows)
    # Drop duplicates by request_id keeping first
    if not df.empty:
        df = df.drop_duplicates(subset=["request_id"], keep="first").reset_index(drop=True)
    return df


def load_answers(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            rid = rec.get("request_id")
            if not rid:
                continue

            model_answers = rec.get("model_answers", {}) or {}
            # Create one row per model answer (Model 1, Model 2)
            for model_label, answer_text in model_answers.items():
                if not answer_text:
                    continue
                rows.append({
                    "request_id": rid,
                    "answer_model_label": model_label,
                    "answer_text": answer_text,
                    "ranker_nickname": rec.get("ranker_nickname"),
                    "asker_nickname": rec.get("asker_nickname"),
                    "difficulty": rec.get("question_difficulty"),
                    # Real model names if available (may be missing)
                    "real_rankings": rec.get("real_rankings"),
                    "real_quality_scores": rec.get("real_quality_scores"),
                })
    df = pd.DataFrame(rows)
    return df


def read_talk_transcripts() -> Dict[str, str]:
    # Map TalkX -> text
    out: Dict[str, str] = {}
    base = os.path.dirname(__file__)
    for i in range(1, 20):  # support up to Talk19.txt if present
        path = os.path.join(base, f"Talk{i}.txt")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    out[f"Talk{i}"] = f.read()
            except Exception:
                pass
    return out


# ------------------------- Analysis/Plots -------------------------

def embed_text_series(texts: pd.Series, embedder: OpenAIEmbedder, cache: EmbeddingCache, model: str) -> np.ndarray:
    to_embed_idx: List[int] = []
    to_embed_texts: List[str] = []
    cached: Dict[int, List[float]] = {}

    # Iterate by position to avoid dtype issues with arbitrary indices
    for i in range(len(texts)):
        txt = (str(texts.iloc[i]) if texts.iloc[i] is not None else "").strip()
        if not txt:
            cached[i] = []  # will fill later once we know dim
            continue
        v = cache.get(txt, model)
        if v is not None:
            cached[i] = v
        else:
            to_embed_idx.append(i)
            to_embed_texts.append(txt)

    if to_embed_texts:
        new_vecs = embedder.embed_texts(to_embed_texts)
        for idx, vec in zip(to_embed_idx, new_vecs):
            cache.put(str(texts.iloc[idx]), model, vec)
            cached[idx] = vec

    # Determine dimension
    dim = 0
    for v in cached.values():
        if v:
            dim = len(v)
            break

    # Build array in original order
    X = np.zeros((len(texts), dim), dtype=float)
    for i in range(len(texts)):
        v = cached.get(i, [0.0] * dim)
        if len(v) == 0:
            v = [0.0] * dim
        X[i, : len(v)] = np.asarray(v, dtype=float)
    return X


def plot_similarity_heatmap(S: np.ndarray, labels: List[str], title: str, out_path: str) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(S, xticklabels=labels, yticklabels=labels, cmap="viridis", vmin=0, vmax=1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_hist(values: np.ndarray, title: str, out_path: str, bins: int = 40) -> None:
    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=bins, color="#4C72B0", alpha=0.85)
    plt.title(title)
    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_scatter_2d(X2: np.ndarray, hue: List[str], title: str, out_path: str, legend_title: str = "Group") -> None:
    plt.figure(figsize=(7.5, 6))
    df = pd.DataFrame({
        "x": X2[:, 0],
        "y": X2[:, 1],
        "hue": hue,
    })
    sns.scatterplot(data=df, x="x", y="y", hue="hue", palette="tab10", s=60, edgecolor="white", linewidth=0.5)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_dendrogram(X: np.ndarray, labels: List[str], title: str, out_path: str) -> None:
    # Convert to distances 1 - cosine
    S = pairwise_cosine_matrix(X)
    D = 1.0 - S
    # Numerical stability: clip negatives & upper bound
    D = np.clip(D, 0.0, 2.0)
    # Numerical stability
    np.fill_diagonal(D, 0.0)
    # linkage expects condensed distance matrix
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="average")
    plt.figure(figsize=(10, 4))
    dendrogram(Z, labels=labels, leaf_rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def compute_intra_inter_similarity(X: np.ndarray, groups: List[str]) -> pd.DataFrame:
    S = pairwise_cosine_matrix(X)
    g = pd.Series(groups)
    uniq = g.unique()
    rows = []
    for grp in uniq:
        idx = np.where(g == grp)[0]
        if len(idx) < 2:
            continue
        intra_vals = S[np.ix_(idx, idx)]
        # exclude diagonal
        intra = intra_vals[np.triu_indices_from(intra_vals, k=1)]

        other = np.where(g != grp)[0]
        inter = S[np.ix_(idx, other)].ravel()

        rows.append({
            "group": grp,
            "intra_mean": float(np.mean(intra)) if intra.size else np.nan,
            "inter_mean": float(np.mean(inter)) if inter.size else np.nan,
            "n_intra_pairs": int(intra.size),
            "n_inter_pairs": int(inter.size),
        })
    return pd.DataFrame(rows)


def plot_intra_inter_bars(stats_df: pd.DataFrame, title: str, out_path: str) -> None:
    if stats_df.empty:
        return
    df = stats_df.melt(id_vars=["group"], value_vars=["intra_mean", "inter_mean"], var_name="type", value_name="similarity")
    plt.figure(figsize=(9, 4.5))
    sns.barplot(data=df, x="group", y="similarity", hue="type", palette={"intra_mean": "#55a868", "inter_mean": "#c44e52"})
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ------------------------------ Main ------------------------------

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Semantic similarity analysis for PaperLLM Q/A logs")
    parser.add_argument("--embed-model", default=EMBED_MODEL_DEFAULT, help="OpenAI embedding model (default: text-embedding-3-large)")
    parser.add_argument("--max-items", type=int, default=0, help="Limit number of Q/A items (0 = no limit)")
    parser.add_argument("--force", action="store_true", help="Force re-embedding (ignore cache). Not implemented; cache is append-only.")
    args = parser.parse_args()

    ensure_dir(PLOTS_DIR)
    ensure_dir(CACHE_DIR)

    # Load data
    q_df = load_questions(QUESTIONS_LOG)
    a_df = load_answers(ANSWERS_LOG)

    if args.max_items and not q_df.empty:
        keep_ids = set(q_df.request_id.head(args.max_items))
        q_df = q_df[q_df.request_id.isin(keep_ids)].reset_index(drop=True)
        a_df = a_df[a_df.request_id.isin(keep_ids)].reset_index(drop=True)

    # Join answers to questions meta (talk, file, difficulty)
    a_df = a_df.merge(q_df[["request_id", "talk", "file", "difficulty", "asker_nickname"]], on="request_id", how="left", suffixes=("", "_q"))

    # Prepare embeddings
    cache = EmbeddingCache(CACHE_PATH)
    embedder = OpenAIEmbedder(args.embed_model)

    # Questions embeddings
    if q_df.empty:
        print("No questions found after filtering.")
        return
    Xq = embed_text_series(q_df["question"], embedder, cache, args.embed_model)

    # Answers embeddings (each model answer is a row)
    if a_df.empty:
        print("No answers found; proceeding with question-only analysis.")
        Xa = np.zeros((0, Xq.shape[1]))
    else:
        Xa = embed_text_series(a_df["answer_text"], embedder, cache, args.embed_model)

    # -------------------- Questions: similarities --------------------
    Sq = pairwise_cosine_matrix(Xq)
    labels_q = [f"{r.talk or ''}:{i}" for i, r in q_df.iterrows()]
    plot_similarity_heatmap(Sq, labels_q, "Question-to-Question Cosine Similarity", os.path.join(PLOTS_DIR, "questions_similarity_heatmap.png"))

    q_vals = Sq[np.triu_indices_from(Sq, k=1)]
    plot_hist(q_vals, "Distribution of Question Pairwise Similarities", os.path.join(PLOTS_DIR, "questions_similarity_hist.png"))

    # PCA scatter by talk and by asker
    Xq2 = pca_project(Xq, k=2)
    plot_scatter_2d(Xq2, (q_df["talk"].fillna("?").tolist()), "Questions: PCA colored by Talk", os.path.join(PLOTS_DIR, "questions_pca_by_talk.png"), legend_title="Talk")
    plot_scatter_2d(Xq2, (q_df["asker_nickname"].fillna("?").tolist()), "Questions: PCA colored by Asker", os.path.join(PLOTS_DIR, "questions_pca_by_asker.png"), legend_title="Asker")

    # Dendrogram for questions
    plot_dendrogram(Xq, [f"{r.talk or ''}|{r.asker_nickname or ''}" for _, r in q_df.iterrows()], "Questions: Hierarchical Clustering", os.path.join(PLOTS_DIR, "questions_dendrogram.png"))

    # Intra/inter similarity by talk and by asker
    talk_stats_q = compute_intra_inter_similarity(Xq, q_df["talk"].fillna("?").tolist())
    plot_intra_inter_bars(talk_stats_q, "Questions: Intra vs Inter similarity by Talk", os.path.join(PLOTS_DIR, "questions_intra_inter_by_talk.png"))

    asker_stats_q = compute_intra_inter_similarity(Xq, q_df["asker_nickname"].fillna("?").tolist())
    plot_intra_inter_bars(asker_stats_q, "Questions: Intra vs Inter similarity by Asker", os.path.join(PLOTS_DIR, "questions_intra_inter_by_asker.png"))

    # --------------------- Answers: similarities ---------------------
    if Xa.shape[0] > 0:
        Sa = pairwise_cosine_matrix(Xa)
        a_vals = Sa[np.triu_indices_from(Sa, k=1)]
        plot_hist(a_vals, "Distribution of Answer Pairwise Similarities (all answers)", os.path.join(PLOTS_DIR, "answers_similarity_hist.png"))

        # Per-question similarity between Model 1 and Model 2
        sims_m1_m2: List[float] = []
        for rid, grp in a_df.groupby("request_id"):
            if grp.shape[0] < 2:
                continue
            # Expect 2 rows (Model 1 & 2); handle more robustly by taking first two
            idxs = grp.index.tolist()[:2]
            v1 = Xa[a_df.index.get_loc(idxs[0])]
            v2 = Xa[a_df.index.get_loc(idxs[1])]
            sims_m1_m2.append(cosine_similarity(v1, v2))
        if sims_m1_m2:
            plot_hist(np.array(sims_m1_m2), "Similarity between paired answers (Model1 vs Model2)", os.path.join(PLOTS_DIR, "answers_pair_similarity_hist.png"))

        # PCA for answers by talk and by asker
        Xa2 = pca_project(Xa, k=2)
        plot_scatter_2d(Xa2, (a_df["talk"].fillna("?").tolist()), "Answers: PCA colored by Talk", os.path.join(PLOTS_DIR, "answers_pca_by_talk.png"), legend_title="Talk")
        plot_scatter_2d(Xa2, (a_df["asker_nickname"].fillna("?").tolist()), "Answers: PCA colored by Asker", os.path.join(PLOTS_DIR, "answers_pca_by_asker.png"), legend_title="Asker")

        # Dendrogram for answers
        plot_dendrogram(Xa, [f"{r.talk or ''}|{r.asker_nickname or ''}|{r.answer_model_label}" for _, r in a_df.iterrows()], "Answers: Hierarchical Clustering", os.path.join(PLOTS_DIR, "answers_dendrogram.png"))

        # Intra/inter similarity by talk and by asker
        talk_stats_a = compute_intra_inter_similarity(Xa, a_df["talk"].fillna("?").tolist())
        plot_intra_inter_bars(talk_stats_a, "Answers: Intra vs Inter similarity by Talk", os.path.join(PLOTS_DIR, "answers_intra_inter_by_talk.png"))

        asker_stats_a = compute_intra_inter_similarity(Xa, a_df["asker_nickname"].fillna("?").tolist())
        plot_intra_inter_bars(asker_stats_a, "Answers: Intra vs Inter similarity by Asker", os.path.join(PLOTS_DIR, "answers_intra_inter_by_asker.png"))

    print(f"Saved plots to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
