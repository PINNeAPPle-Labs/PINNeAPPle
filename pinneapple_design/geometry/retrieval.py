"""Geometry retrieval: find the parts most similar to a query part.

Use: before training a surrogate for a new part, look for a similar part that already has one
(transfer / fine-tune instead of starting from scratch), or deduplicate a CAD library.

Two sources of vectors, one index:

- :func:`shape_descriptor` -- no ML, works offline on any mesh: the D2 shape distribution
  (histogram of distances between random surface point pairs, normalised by their mean; Osada,
  Funkhouser, Chazelle & Dobkin, "Shape distributions", ACM Trans. Graph. 21(4) (2002) 807-832),
  concatenated with sorted principal extents. Invariant to rotation, translation and scale.
- precomputed embeddings, e.g. the ~1M ABC parts rendered, captioned by a VLM and text-embedded
  by finalrev ("Embedding one million 3D models", 2026; HF ``daveferbear/3d-model-images-embeddings``,
  no license declared, ABC/Onshape origin -> research only, see ``pinneapple_catalog``):
  :func:`load_caption_index` builds an index over its parquet shards, with keyword search on the
  captions and neighbour search between parts. Its embedding model is not published, so new text
  queries cannot be embedded into that space; keyword search on captions is used instead.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def shape_descriptor(mesh, n_pairs: int = 20000, bins: int = 48, seed: int = 0) -> np.ndarray:
    """D2 histogram (``bins``) + 3 sorted normalised principal extents, L2-normalised."""
    rng = np.random.default_rng(seed)
    pts, _ = _sample_surface(mesh, 2 * n_pairs, rng)
    d = np.linalg.norm(pts[:n_pairs] - pts[n_pairs:], axis=1)
    d = d / (d.mean() + 1e-12)
    hist, _ = np.histogram(d, bins=bins, range=(0.0, 3.0))
    hist = hist / hist.sum()
    c = pts - pts.mean(0)
    ev = np.sort(np.sqrt(np.maximum(np.linalg.eigvalsh(c.T @ c / len(c)), 0)))[::-1]
    ext = ev / (ev[0] + 1e-12)
    v = np.concatenate([hist, 0.5 * ext])
    return v / (np.linalg.norm(v) + 1e-12)


def _sample_surface(mesh, n, rng):
    v = np.asarray(mesh.vertices, float)
    f = np.asarray(mesh.faces, int)
    tri = v[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    idx = rng.choice(len(f), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    p = (1 - s)[:, None] * tri[idx, 0] + (s * (1 - r2))[:, None] * tri[idx, 1] + (s * r2)[:, None] * tri[idx, 2]
    return p, idx


@dataclass
class GeometryIndex:
    """Cosine-similarity index over part vectors (brute force; fine up to ~10^5-10^6 parts)."""

    ids: List[str] = field(default_factory=list)
    vectors: Optional[np.ndarray] = None
    captions: Dict[str, str] = field(default_factory=dict)

    def add(self, part_id: str, vector, caption: Optional[str] = None) -> None:
        v = np.asarray(vector, np.float32).reshape(1, -1)
        v = v / (np.linalg.norm(v) + 1e-12)
        self.vectors = v if self.vectors is None else np.vstack([self.vectors, v])
        self.ids.append(str(part_id))
        if caption:
            self.captions[str(part_id)] = caption

    def add_mesh(self, part_id: str, mesh, **kw) -> None:
        self.add(part_id, shape_descriptor(mesh, **kw))

    def search(self, vector, k: int = 5, exclude: Iterable[str] = ()) -> List[Tuple[str, float]]:
        if self.vectors is None:
            return []
        q = np.asarray(vector, np.float32).ravel()
        sims = self.vectors @ (q / (np.linalg.norm(q) + 1e-12))
        skip = set(exclude)
        out = []
        for i in np.argsort(-sims):
            if self.ids[i] not in skip:
                out.append((self.ids[i], float(sims[i])))
            if len(out) == k:
                break
        return out

    def neighbours(self, part_id: str, k: int = 5) -> List[Tuple[str, float]]:
        return self.search(self.vectors[self.ids.index(str(part_id))], k, exclude=[str(part_id)])

    def search_captions(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Keyword search (TF-IDF-weighted term overlap) over captions."""
        terms = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 1]
        if not terms or not self.captions:
            return []
        docs = {pid: re.findall(r"[a-z0-9]+", c.lower()) for pid, c in self.captions.items()}
        n = len(docs)
        idf = {t: np.log((n + 1) / (1 + sum(t in set(toks) for toks in docs.values()))) + 1 for t in terms}
        scores = []
        for pid, toks in docs.items():
            if not toks:
                continue
            s = sum(idf[t] * toks.count(t) / len(toks) for t in terms)
            if s > 0:
                scores.append((pid, float(s)))
        return sorted(scores, key=lambda x: -x[1])[:k]


def load_caption_index(parquet_paths: Sequence[str], id_column: str = "abc_id", limit: Optional[int] = None) -> GeometryIndex:
    """Index finalrev-style parquet shards (columns ``abc_id``, ``caption``, ``embedding``)."""
    import pyarrow.parquet as pq

    idx = GeometryIndex()
    for p in parquet_paths:
        t = pq.read_table(p, columns=[id_column, "caption", "embedding"])
        for pid, cap, emb in zip(t.column(id_column).to_pylist(), t.column("caption").to_pylist(),
                                 t.column("embedding").to_pylist()):
            idx.add(pid, emb, cap)
            if limit and len(idx.ids) >= limit:
                return idx
    return idx
