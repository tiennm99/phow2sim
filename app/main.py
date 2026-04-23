"""FastAPI entry point for phow2sim.

Read-only endpoints over PhoW2V: pairwise similarity, nearest neighbors,
vocab lookup, random word. Stateless. Response shapes mirror word2sim so
callers can swap the two services via URL alone.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from app.vectors import canonicalize, load_model, neighbors, random_word, similarity


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Force download + model load before accepting traffic (PhoW2V word-300d ~60s cold).
    load_model()
    yield


app = FastAPI(title="phow2sim", version="0.1.0", lifespan=lifespan)


class SimilarityResponse(BaseModel):
    a: str
    b: str
    canonical_a: str | None
    canonical_b: str | None
    in_vocab_a: bool
    in_vocab_b: bool
    similarity: float | None  # null iff either side is out-of-vocab


class NeighborEntry(BaseModel):
    word: str
    similarity: float


class NeighborsResponse(BaseModel):
    word: str
    canonical: str | None
    in_vocab: bool
    neighbors: list[NeighborEntry]


class VocabResponse(BaseModel):
    word: str
    canonical: str | None
    in_vocab: bool


class RandomWordResponse(BaseModel):
    word: str
    rank: int


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/similarity", response_model=SimilarityResponse)
def get_similarity(
    a: str = Query(min_length=1, max_length=64),
    b: str = Query(min_length=1, max_length=64),
) -> SimilarityResponse:
    kv = load_model()
    canonical_a = canonicalize(kv, a)
    canonical_b = canonicalize(kv, b)
    score = (
        similarity(kv, canonical_a, canonical_b)
        if canonical_a and canonical_b
        else None
    )
    return SimilarityResponse(
        a=a,
        b=b,
        canonical_a=canonical_a,
        canonical_b=canonical_b,
        in_vocab_a=canonical_a is not None,
        in_vocab_b=canonical_b is not None,
        similarity=score,
    )


@app.get("/neighbors", response_model=NeighborsResponse)
def get_neighbors(
    word: str = Query(min_length=1, max_length=64),
    topn: int = Query(default=10, ge=1, le=1000),
) -> NeighborsResponse:
    kv = load_model()
    canonical = canonicalize(kv, word)
    if canonical is None:
        return NeighborsResponse(
            word=word, canonical=None, in_vocab=False, neighbors=[]
        )
    entries = [
        NeighborEntry(word=w, similarity=s) for w, s in neighbors(kv, canonical, topn)
    ]
    return NeighborsResponse(
        word=word, canonical=canonical, in_vocab=True, neighbors=entries
    )


@app.get("/vocab", response_model=VocabResponse)
def get_vocab(word: str = Query(min_length=1, max_length=64)) -> VocabResponse:
    kv = load_model()
    canonical = canonicalize(kv, word)
    return VocabResponse(word=word, canonical=canonical, in_vocab=canonical is not None)


@app.get("/random", response_model=RandomWordResponse)
def get_random(
    min_rank: int = Query(default=100, ge=0),
    max_rank: int = Query(default=50000, ge=1),
    alpha_only: bool = Query(default=True),
    min_len: int = Query(default=2, ge=1, le=64),
    max_len: int = Query(default=20, ge=1, le=64),
) -> RandomWordResponse:
    if min_rank >= max_rank:
        raise HTTPException(status_code=400, detail="min_rank must be < max_rank")
    if min_len > max_len:
        raise HTTPException(status_code=400, detail="min_len must be <= max_len")
    kv = load_model()
    word = random_word(
        kv,
        min_rank=min_rank,
        max_rank=max_rank,
        alpha_only=alpha_only,
        min_len=min_len,
        max_len=max_len,
    )
    if word is None:
        raise HTTPException(
            status_code=503, detail="no word matched filter; loosen the constraints"
        )
    return RandomWordResponse(word=word, rank=kv.key_to_index[word])
