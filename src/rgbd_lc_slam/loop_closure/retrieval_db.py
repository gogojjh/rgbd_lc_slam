from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .types import RetrievalCandidate
from .utils import l2_normalize


def _try_import_faiss():
    try:
        import faiss  # type: ignore

        return faiss
    except Exception:
        return None


@dataclass
class RetrievalDatabaseConfig:
    dim: int
    use_faiss: bool = True
    metric: str = "ip"  # "ip" for cosine/inner-product on normalized vectors


class RetrievalDatabase:
    """Descriptor DB for loop closure retrieval.

    Stores L2-normalized float32 descriptors.

    If FAISS is available it will be used; otherwise brute-force numpy search.
    """

    def __init__(self, cfg: RetrievalDatabaseConfig):
        self.cfg = cfg
        self.faiss = _try_import_faiss() if cfg.use_faiss else None

        self._ids: List[int] = []
        self._desc: List[np.ndarray] = []

        self._index = None
        if self.faiss is not None:
            if cfg.metric == "ip":
                self._index = self.faiss.IndexFlatIP(cfg.dim)
            else:
                self._index = self.faiss.IndexFlatL2(cfg.dim)

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, frame_id: int, desc: np.ndarray) -> None:
        desc = np.asarray(desc, dtype=np.float32).reshape(-1)
        if desc.shape[0] != self.cfg.dim:
            raise ValueError(f"Descriptor dim mismatch: expected {self.cfg.dim}, got {desc.shape[0]}")
        desc = l2_normalize(desc).astype(np.float32)

        self._ids.append(int(frame_id))
        self._desc.append(desc)

        if self._index is not None:
            self._index.add(desc[None, :])

    def query(
        self,
        desc: np.ndarray,
        top_k: int = 10,
        *,
        exclude_ids: Optional[Sequence[int]] = None,
    ) -> List[RetrievalCandidate]:
        if len(self) == 0:
            return []

        q = np.asarray(desc, dtype=np.float32).reshape(1, -1)
        q = l2_normalize(q).astype(np.float32)

        exclude = set(int(x) for x in (exclude_ids or []))

        if self._index is not None:
            scores, idx = self._index.search(q, min(top_k + len(exclude), len(self)))
            scores = scores.reshape(-1)
            idx = idx.reshape(-1)
            out: List[RetrievalCandidate] = []
            for s, k in zip(scores, idx):
                if k < 0:
                    continue
                fid = self._ids[int(k)]
                if fid in exclude:
                    continue
                out.append(RetrievalCandidate(frame_id=fid, score=float(s)))
                if len(out) >= top_k:
                    break
            return out

        # brute force
        D = np.stack(self._desc, axis=0)  # (N,dim)
        if self.cfg.metric == "ip":
            scores = (D @ q.reshape(-1)).astype(np.float32)
        else:
            # negative L2 for sorting descending (larger is better)
            scores = (-np.sum((D - q) ** 2, axis=1)).astype(np.float32)

        order = np.argsort(-scores)
        out: List[RetrievalCandidate] = []
        for k in order:
            fid = self._ids[int(k)]
            if fid in exclude:
                continue
            out.append(RetrievalCandidate(frame_id=fid, score=float(scores[int(k)])))
            if len(out) >= top_k:
                break
        return out
