"""Unsupervised exploration of archaeology embedding clusters.

Uses vectors stored in the local Milvus index. Reduces to 2D with PCA + t-SNE,
clusters with KMeans, and writes a CSV + cluster summary.

Usage:
    source .env
    uv run python scripts/explore_embeddings.py

Outputs:
    archaeology_model/indices/exploration_clusters.csv
    archaeology_model/indices/exploration_clusters.json
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from pymilvus import MilvusClient
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

COLLECTION_NAME = "archaeology_chunks"
DB_PATH = "archaeology_model/indices/milvus.db"
OUTPUT_DIR = Path("archaeology_model/indices")
N_CLUSTERS = 20
RANDOM_STATE = 42


def load_vectors(limit: int = 20000) -> tuple[np.ndarray, list[dict]]:
    milvus = MilvusClient(uri=DB_PATH)
    milvus.load_collection(COLLECTION_NAME)

    results = milvus.query(
        collection_name=COLLECTION_NAME,
        output_fields=[
            "vector",
            "title",
            "heading",
            "text",
            "source_file",
            "region",
            "period",
            "corpus",
        ],
        limit=limit,
    )
    vectors = np.array([r["vector"] for r in results], dtype=np.float32)
    return vectors, results


def summarize_cluster(metas: list[dict]) -> dict:
    """Return representative samples and common source files for a cluster."""
    titles = [m.get("title", "") for m in metas if m.get("title")]
    headings = [m.get("heading", "") for m in metas if m.get("heading")]
    sources = [m.get("source_file", "") for m in metas if m.get("source_file")]
    texts = [m.get("text", "")[:300] for m in metas[:5]]
    return {
        "size": len(metas),
        "top_titles": _top_freq(titles, 5),
        "top_headings": _top_freq(headings, 5),
        "top_sources": _top_freq(sources, 3),
        "samples": texts,
    }


def _top_freq(items: list[str], k: int) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return sorted(counts.items(), key=lambda x: -x[1])[:k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore archaeology embedding clusters.")
    parser.add_argument("--clusters", "-n", type=int, default=N_CLUSTERS, help="Number of KMeans clusters")
    parser.add_argument("--limit", type=int, default=20000, help="Max vectors to load")
    args = parser.parse_args()

    n_clusters = args.clusters
    print("Loading vectors from Milvus...")
    vectors, metas = load_vectors(limit=args.limit)
    print(f"Loaded {len(vectors)} vectors of dimension {vectors.shape[1]}")

    print(f"Clustering into {n_clusters} groups with KMeans...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init="auto")
    clusters = kmeans.fit_predict(vectors)

    print("Reducing to 2D with PCA + t-SNE...")
    pca = PCA(n_components=50, random_state=RANDOM_STATE)
    reduced = pca.fit_transform(vectors)
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, init="pca")
    coords = tsne.fit_transform(reduced)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "exploration_clusters.csv"
    json_path = OUTPUT_DIR / "exploration_clusters.json"

    rows = []
    for i, meta in enumerate(metas):
        rows.append(
            {
                "id": i,
                "cluster": int(clusters[i]),
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "title": meta.get("title", ""),
                "heading": meta.get("heading", ""),
                "region": meta.get("region", ""),
                "period": meta.get("period", ""),
                "corpus": meta.get("corpus", ""),
                "source_file": meta.get("source_file", ""),
                "text_preview": meta.get("text", "")[:300],
            }
        )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")

    cluster_groups: dict[int, list[dict]] = {}
    for i, c in enumerate(clusters):
        cluster_groups.setdefault(int(c), []).append(metas[i])

    summary = {c: summarize_cluster(ms) for c, ms in cluster_groups.items()}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Wrote {json_path}")

    print("\nCluster summary:")
    for c in sorted(summary):
        s = summary[c]
        print(f"\nCluster {c} ({s['size']} chunks)")
        print(f"  top titles: {[t[0] for t in s['top_titles'][:3]]}")
        print(f"  top headings: {[h[0] for h in s['top_headings'][:3]]}")
        print(f"  sample: {s['samples'][0][:120]}...")


if __name__ == "__main__":
    main()
