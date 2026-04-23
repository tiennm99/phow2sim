# phow2sim

Tiny HTTP service that returns Vietnamese word2vec similarity and nearest
neighbors — Vietnamese sibling of [`word2sim`](../word2sim). Same endpoint
shapes; swap URLs and it's a drop-in replacement.

Backed by [**PhoW2V**](https://github.com/VinAIResearch/PhoW2V) (VinAI), the
largest pretrained Vietnamese word vectors available. Chosen over PhoBERT
for this purpose because word2vec's similarity distribution is wide
enough to drive a Semantle-style warmth meter, whereas raw transformer
embeddings saturate at the top.

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

```bash
docker compose up --build
# First boot downloads ~1.2GB (word-300d) into the `phow2v-cache` volume.
# Model parse ~60s. A binary cache is written on first success so later
# restarts take ~10s.
```

Health check start period is 10 min to cover the download + parse.

## Switching variant

Edit `docker-compose.yml` or pass env:

```bash
MODEL_URL=https://public.vinai.io/word2vec_vi_syllables_100dims.zip \
MODEL_PATH=/data/phow2v/word2vec_vi_syllables_100dims.txt \
MODEL_VARIANT=syllable \
docker compose up --build
```

Delete the `phow2v-cache` volume when switching, otherwise the stale
`.bin` from the previous variant will load instead.

## Manual model population

Skip the auto-download if you want to prepare the volume ahead of time:

```bash
./scripts/download-phow2v.sh word 300        # word-300d into ./models
# then mount ./models as /data/phow2v in docker-compose.yml
```

## Config (env vars)

| Var | Default | Meaning |
|---|---|---|
| `MODEL_URL` | `https://public.vinai.io/word2vec_vi_words_300dims.zip` | fetched on first boot if `MODEL_PATH` absent |
| `MODEL_PATH` | `/data/phow2v/word2vec_vi_words_300dims.txt` | where the text-format vectors live |
| `MODEL_VARIANT` | `word` | declarative hint for the caller; `word` or `syllable` |

## Using from doantu (miti99bot)

The Cloudflare Worker module's `api-client.js` already produces the same
response shape. Replace `embedPair` + local cosine with a single `fetch`:

```js
const url = `${env.PHOW2SIM_URL}/similarity?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`;
const resp = await fetch(url, { headers: { Authorization: `Bearer ${env.PHOW2SIM_TOKEN}` } });
return await resp.json();  // { in_vocab_a, in_vocab_b, similarity, ... }
```

Auth is **not** built-in here — add a reverse proxy (Caddy, Cloudflare
Tunnel, or nginx) in front that checks a bearer token before passing
through. The service itself trusts its caller.

## Project layout

```
phow2sim/
├── app/
│   ├── main.py       # FastAPI routes
│   └── vectors.py    # PhoW2V loader + canonicalize + similarity/neighbors/random
├── scripts/
│   └── download-phow2v.sh
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Credits

- Vectors: [PhoW2V](https://github.com/VinAIResearch/PhoW2V) by VinAI Research (research license — see their repo).
- API shape: sibling of [`word2sim`](../word2sim).
