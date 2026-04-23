# phow2sim

Tiny HTTP service that returns Vietnamese word2vec similarity and nearest
neighbors — Vietnamese sibling of [`word2sim`](../word2sim). Same endpoint
shapes; swap URLs and it's a drop-in replacement.

Backed by [**PhoW2V**](https://github.com/datquocnguyen/PhoW2V) (VinAI /
Dat Quoc Nguyen), the largest pretrained Vietnamese word vectors
available. Chosen over PhoBERT for this purpose because word2vec's
similarity distribution is wide enough to drive a Semantle-style warmth
meter, whereas raw transformer embeddings saturate at the top.

> **License note.** PhoW2V's research-only license forbids public
> redistribution, so this service doesn't — and can't — embed or
> auto-download the vectors from any public URL. You supply your own
> private mirror (typically a Nextcloud instance you control). See
> [Quick start](#quick-start).

## Stack

- FastAPI + uvicorn
- gensim (loads PhoW2V `.txt` files; caches a binary `.bin` alongside for 5× faster restarts)

## Variants

PhoW2V ships in four flavors. Pick one per deployment.

| Variant | Dims | Size | Best for |
|---|---|---|---|
| `word-100`    | 100 | ~400MB | low-RAM hosts, compound-aware |
| `word-300`    | 300 | ~1.2GB | **default** — best quality, compound-aware |
| `syllable-100`| 100 | ~50MB  | single-syllable guesses, tiny footprint |
| `syllable-300`| 300 | ~150MB | single-syllable guesses, richer vectors |

The "word" variants expect underscore-joined compounds (`sinh_viên`);
the "syllable" variants have no multi-token keys. The canonicalizer
tries both forms, but the client should pre-segment for the word variant
if it wants reliable coverage of compounds.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness probe |
| GET | `/similarity?a=X&b=Y` | cosine similarity between two keys |
| GET | `/neighbors?word=X&topn=10` | nearest-neighbor keys with scores |
| GET | `/vocab?word=X` | check in-vocab; return canonical form |
| GET | `/random` | random vocab key, filtered for game-friendliness |

Response shape is identical to word2sim.

### Examples

```bash
curl 'http://localhost:8001/similarity?a=con_chó&b=con_mèo'
# {"a":"con_chó","b":"con_mèo","canonical_a":"con_chó","canonical_b":"con_mèo",
#  "in_vocab_a":true,"in_vocab_b":true,"similarity":0.78}

curl 'http://localhost:8001/neighbors?word=đại_học&topn=5'

curl 'http://localhost:8001/vocab?word=con%20ch%C3%B3'   # "con chó" → tries "con_chó"
# {"word":"con chó","canonical":"con_chó","in_vocab":true}

curl 'http://localhost:8001/random?min_rank=500&max_rank=20000&min_len=3&max_len=12'
```

Out-of-vocab returns `in_vocab:false` and `similarity:null`. Lookup
tries exact → lowercase → space-to-underscore variants.

## Quick start

1. **Get the vectors once.** Download from the [upstream Google Drive
   mirror](https://drive.google.com/drive/folders/1NZhZFYbcwKzLpvvGdJUdPbwEVdVW4E3j?usp=drive_link)
   (the one linked from the PhoW2V README — the original
   `public.vinai.io` URLs are dead). Four zips; keep the one matching
   your chosen variant.

2. **Host the zip somewhere a plain `GET` can reach it.** Options:
   - **Nextcloud public share** with file upload, then use the
     `/download` endpoint: `https://cloud.example.com/s/<token>/download`.
     The share token acts as the capability; leave it unguessable and
     unlisted.
   - Any signed/pre-signed URL from your object store (S3, R2,
     BackBlaze B2), or your own HTTP(S) endpoint.

   The service sends **no auth headers** — any authentication must be
   baked into the URL itself. This keeps the code minimal and puts
   hosting policy on the operator.

3. **Configure env.** Copy `.env.example` to `.env` and set `MODEL_URL`:
   ```bash
   cp .env.example .env
   # edit .env:
   #   MODEL_URL=https://cloud.example.com/s/abc123XYZ/download
   ```

4. **Boot.**
   ```bash
   docker compose up --build
   ```
   First boot streams ~1.2GB (word-300d) into the `phow2v-cache` volume,
   then parses ~60s. A binary `.bin` is written alongside so later
   restarts load in ~10s. Health check start period is 10 min to cover
   the first-boot cost.

### Alternative: mount a local file instead

If you've already downloaded the `.txt` locally and don't want to
re-upload anywhere, skip `MODEL_URL` entirely and mount the file. In
`docker-compose.yml`, uncomment the bind mount:

```yaml
volumes:
  - phow2v-cache:/data/phow2v
  - ./models/word2vec_vi_words_300dims.txt:/data/phow2v/word2vec_vi_words_300dims.txt:ro
```

Then `docker compose up` boots straight into parse — no download step.

## Switching variant

Host the desired zip and update `.env`:

```bash
MODEL_URL=https://cloud.example.com/s/<token-for-syllables-100>/download
MODEL_PATH=/data/phow2v/word2vec_vi_syllables_100dims.txt
```

Delete the `phow2v-cache` volume when switching, otherwise the stale
`.bin` from the previous variant loads instead:

```bash
docker compose down -v && docker compose up --build
```

## Config (env vars)

| Var | Default | Meaning |
|---|---|---|
| `MODEL_URL` | `""` | URL that serves the zip via a plain GET. Bake any auth into the URL. Optional if `MODEL_PATH` is already populated via a bind mount. |
| `MODEL_PATH` | `/data/phow2v/word2vec_vi_words_300dims.txt` | Where the text-format vectors live. A binary `.bin` sibling is written on first parse. |

## Auth

The service does not authenticate its callers. Put it behind a reverse
proxy (Caddy, nginx, Cloudflare Tunnel) if you need access control.

## Project layout

```
phow2sim/
├── app/
│   ├── main.py       # FastAPI routes
│   └── vectors.py    # PhoW2V loader + canonicalize + similarity/neighbors/random
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example      # copy to .env and set MODEL_URL
```

## Credits

- Vectors: [PhoW2V](https://github.com/datquocnguyen/PhoW2V) by Dat Quoc Nguyen / VinAI Research (research-only license — see upstream).
- API shape: sibling of [`word2sim`](../word2sim).
