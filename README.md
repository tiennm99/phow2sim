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
   `public.vinai.io` URLs are dead). You'll get four zips; keep the one
   matching your chosen variant.

2. **Upload the zip to your Nextcloud** in a folder like `phow2v/`. In
   Nextcloud → Settings → Security, generate an **app password** for
   this service (do not use your real login password, and do not disable
   2FA if you use it).

3. **Configure env.** Copy `.env.example` to `.env` and fill in:
   ```bash
   cp .env.example .env
   # then edit .env:
   #   MODEL_URL=https://cloud.example.com/remote.php/dav/files/<user>/phow2v/word2vec_vi_words_300dims.zip
   #   MODEL_DOWNLOAD_USER=<nextcloud-username>
   #   MODEL_DOWNLOAD_PASSWORD=<app-password>
   ```

4. **Boot.**
   ```bash
   docker compose up --build
   ```
   First boot streams ~1.2GB (word-300d) from Nextcloud into the
   `phow2v-cache` volume, then parses ~60s. A binary `.bin` is written
   alongside so later restarts load in ~10s. Health check start period
   is 10 min to cover the first-boot cost.

### Using a public share instead of WebDAV

If you'd rather create a password-protected Nextcloud share link:

```
MODEL_URL=https://cloud.example.com/s/<shareToken>/download
MODEL_DOWNLOAD_USER=
MODEL_DOWNLOAD_PASSWORD=<share-password>
```

Basic auth with an empty username is how Nextcloud authenticates a
public-share password.

## Switching variant

Upload the desired zip to Nextcloud, then update `.env`:

```bash
MODEL_URL=https://cloud.example.com/remote.php/dav/files/<user>/phow2v/word2vec_vi_syllables_100dims.zip
MODEL_PATH=/data/phow2v/word2vec_vi_syllables_100dims.txt
MODEL_VARIANT=syllable
```

Delete the `phow2v-cache` volume when switching, otherwise the stale
`.bin` from the previous variant will load instead:

```bash
docker compose down -v && docker compose up --build
```

## Config (env vars)

| Var | Default | Meaning |
|---|---|---|
| `MODEL_URL` | *(required)* | Private URL to the PhoW2V zip. WebDAV or Nextcloud public-share `/download` URL. |
| `MODEL_DOWNLOAD_USER` | `""` | Basic-auth user. Empty for Nextcloud public-share password auth. |
| `MODEL_DOWNLOAD_PASSWORD` | *(required)* | Basic-auth password — Nextcloud app password, or share password. |
| `MODEL_PATH` | `/data/phow2v/word2vec_vi_words_300dims.txt` | Where the text-format vectors are persisted. |
| `MODEL_VARIANT` | `word` | `word` or `syllable`. Declarative hint; must match the file you uploaded. |

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
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example      # copy to .env and fill in Nextcloud creds
```

## Credits

- Vectors: [PhoW2V](https://github.com/datquocnguyen/PhoW2V) by Dat Quoc Nguyen / VinAI Research (research-only license — see upstream).
- API shape: sibling of [`word2sim`](../word2sim).
