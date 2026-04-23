"""PhoW2V model loader and similarity primitives.

PhoW2V (VinAI) ships as word2vec-text format (.txt), available in four
variants: word/syllable × 100/300 dims. Text format is slow to parse on
first boot (~30-60s for the 300d word model), so we cache a binary .bin
alongside the .txt after the first successful load — subsequent starts
use the fast binary path.

Tokenization matters. The "word" variant expects underscore-joined
compounds ("sinh_viên"); the "syllable" variant expects single syllables
("sinh", "viên"). Callers must normalize to match before querying.

Model source: PhoW2V's research license forbids public redistribution,
so MODEL_URL is expected to point at a private, auth-gated mirror
(typically Nextcloud WebDAV). HTTP Basic auth is applied when
MODEL_DOWNLOAD_USER / MODEL_DOWNLOAD_PASSWORD are set.
"""

from __future__ import annotations

import base64
import os
import random as _random
import unicodedata
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

from gensim.models import KeyedVectors

_MODEL: Optional[KeyedVectors] = None
_DOWNLOAD_CHUNK = 1 << 20  # 1 MiB; keeps peak RAM flat for ~1GB downloads.


def _build_request(url: str) -> urllib.request.Request:
    """Build a GET with optional Basic auth from env. WebDAV-friendly."""
    req = urllib.request.Request(url)
    user = os.environ.get("MODEL_DOWNLOAD_USER", "")
    password = os.environ.get("MODEL_DOWNLOAD_PASSWORD")
    # Password-only (empty user) covers Nextcloud public-share passwords too.
    if password is not None:
        creds = base64.b64encode(f"{user}:{password}".encode()).decode("ascii")
        req.add_header("Authorization", f"Basic {creds}")
    return req


def _download_and_extract(url: str, target_txt: Path) -> None:
    """Fetch a PhoW2V zip (streamed) and extract its .txt into target_txt."""
    target_txt.parent.mkdir(parents=True, exist_ok=True)
    zip_path = target_txt.with_suffix(".zip")

    req = _build_request(url)
    with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as dst:
        while True:
            chunk = resp.read(_DOWNLOAD_CHUNK)
            if not chunk:
                break
            dst.write(chunk)

    with zipfile.ZipFile(zip_path) as zf:
        txt_members = [m for m in zf.namelist() if m.endswith(".txt")]
        if not txt_members:
            raise RuntimeError(f"no .txt file inside {url}")
        # Flatten into target_txt regardless of archive's internal layout.
        with zf.open(txt_members[0]) as src, open(target_txt, "wb") as dst:
            while True:
                chunk = src.read(_DOWNLOAD_CHUNK)
                if not chunk:
                    break
                dst.write(chunk)
    zip_path.unlink(missing_ok=True)


def load_model() -> KeyedVectors:
    """Return the singleton KeyedVectors, loading (and downloading) on first call."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    txt_path = Path(os.environ["MODEL_PATH"])
    bin_cache = txt_path.with_suffix(".bin")

    # Prefer the cached binary form for ~5x faster cold start.
    if bin_cache.exists():
        _MODEL = KeyedVectors.load_word2vec_format(str(bin_cache), binary=True)
        return _MODEL

    if not txt_path.exists():
        url = os.environ.get("MODEL_URL")
        if not url:
            raise FileNotFoundError(
                f"MODEL_PATH {txt_path} missing and no MODEL_URL set for auto-download"
            )
        _download_and_extract(url, txt_path)

    _MODEL = KeyedVectors.load_word2vec_format(str(txt_path), binary=False)
    # Persist the fast-load cache next to the source .txt.
    try:
        _MODEL.save_word2vec_format(str(bin_cache), binary=True)
    except OSError:
        pass  # Read-only volume is fine; we'll just pay the txt-parse cost again.
    return _MODEL


def _variant_candidates(word: str) -> list[str]:
    """Casing/segmentation candidates ordered by specificity.

    PhoW2V-word uses underscores for compounds; PhoW2V-syllable has no
    multi-token entries. Trying both forms covers either config without
    caller branching.
    """
    stripped = word.strip()
    lowered = stripped.lower()
    joined = stripped.replace(" ", "_")
    joined_lower = lowered.replace(" ", "_")
    # Ordered, de-duplicated.
    seen: set[str] = set()
    out: list[str] = []
    for c in (stripped, lowered, joined, joined_lower):
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def canonicalize(kv: KeyedVectors, word: str) -> Optional[str]:
    """Resolve `word` to its in-vocab form, trying exact → lower → space→underscore."""
    for candidate in _variant_candidates(word):
        if candidate in kv:
            return candidate
    return None


def similarity(kv: KeyedVectors, a: str, b: str) -> float:
    """Cosine similarity between two in-vocab keys. Caller must canonicalize."""
    return float(kv.similarity(a, b))


def neighbors(kv: KeyedVectors, word: str, topn: int) -> list[tuple[str, float]]:
    """Top-N nearest-neighbor keys with cosine scores. Caller must canonicalize."""
    return [(w, float(s)) for w, s in kv.most_similar(word, topn=topn)]


def _is_vietnamese_wordlike(word: str) -> bool:
    """Reject digits and punctuation; accept Latin letters, Vietnamese diacritics, `_`."""
    for ch in word:
        if ch == "_":
            continue
        cat = unicodedata.category(ch)
        # Ll/Lu = letters, Mn = combining marks (diacritics on decomposed input).
        if cat not in ("Ll", "Lu", "Lo", "Lt", "Mn"):
            return False
    return True


def random_word(
    kv: KeyedVectors,
    *,
    min_rank: int = 0,
    max_rank: Optional[int] = None,
    alpha_only: bool = True,
    min_len: int = 1,
    max_len: int = 64,
    max_attempts: int = 1000,
) -> Optional[str]:
    """Return a random vocab key matching filters, or None within attempt budget.

    `index_to_key` is frequency-ordered for word2vec-text files, so rank bounds
    behave as a frequency window. `alpha_only=True` accepts Vietnamese letters
    and the word-boundary `_` — rejects numerals, punctuation, and foreign
    scripts that sometimes leak into Vietnamese corpora.
    """
    vocab = kv.index_to_key
    upper = min(max_rank, len(vocab)) if max_rank is not None else len(vocab)
    if min_rank >= upper:
        return None
    for _ in range(max_attempts):
        word = vocab[_random.randrange(min_rank, upper)]
        if not (min_len <= len(word) <= max_len):
            continue
        if alpha_only and not _is_vietnamese_wordlike(word):
            continue
        return word
    return None
