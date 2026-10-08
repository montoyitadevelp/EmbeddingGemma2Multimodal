<div align="center">

# 🧬 EmbeddingGemma 2 — Multimodal Retrieval

**A single Jupyter notebook showing how Google's open embedding model maps text, images, audio, and video into one 768-dimensional vector space — so any of them can query any of them.**

</div>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://jupyter.org/"><img src="https://img.shields.io/badge/Jupyter-notebook-F37626?logo=jupyter&logoColor=white" alt="Jupyter"></a>
  <a href="https://www.sbert.net/"><img src="https://img.shields.io/badge/sentence__transformers-≥6.1.0-FFD21E?logo=huggingface&logoColor=white" alt="sentence-transformers"></a>
  <a href="https://huggingface.co/google/embeddinggemma-2"><img src="https://img.shields.io/badge/EmbeddingGemma_2-768d-FF6F00?logo=google&logoColor=white" alt="EmbeddingGemma 2"></a>
  <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/HuggingFace-Hub-FFD21E?logo=huggingface&logoColor=white" alt="HuggingFace"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://github.com/pytorch/torchcodec"><img src="https://img.shields.io/badge/torchcodec-video%20decoding-EE4C2C?logo=pytorch&logoColor=white" alt="torchcodec"></a>
  <a href="https://pymupdf.readthedocs.io/"><img src="https://img.shields.io/badge/PyMuPDF-PDF%20→%20image-707070?logo=adobeacrobatreader&logoColor=white" alt="PyMuPDF"></a>
  <a href="https://librosa.org/"><img src="https://img.shields.io/badge/librosa-audio%20loading-FF6F00?logo=python&logoColor=white" alt="librosa"></a>
  <a href="https://pypi.org/project/gTTS/"><img src="https://img.shields.io/badge/gTTS-TTS-4285F4?logo=google&logoColor=white" alt="gTTS"></a>
  <a href="https://numpy.org/"><img src="https://img.shields.io/badge/NumPy-arrays-013243?logo=numpy&logoColor=white" alt="NumPy"></a>
  <a href="https://matplotlib.org/"><img src="https://img.shields.io/badge/Matplotlib-plots-11557C?logo=matplotlib&logoColor=white" alt="Matplotlib"></a>
</p>

<p align="center">
  <em>text ↔ photo ↔ audio ↔ video · all in one 768-D vector space · one cosine similarity · no late fusion</em>
</p>

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Quick Start](#2-quick-start)
3. [The 9 Demos](#3-the-9-demos)
4. [Demo 1 — One Space for Every Modality](#4-demo-1--one-space-for-every-modality)
5. [Demo 2 — Search 1,000 Photos with Text](#5-demo-2--search-1000-photos-with-text)
6. [Demo 3 — Any Language, or a Photo as the Query](#6-demo-3--any-language-or-a-photo-as-the-query)
7. [Demo 4 — Ask with Your Voice](#7-demo-4--ask-with-your-voice)
8. [Demo 5 — Search Sounds with Text](#8-demo-5--search-sounds-with-text)
9. [Demo 6 — Video Moments Finder](#9-demo-6--video-moments-finder)
10. [Demo 7 — Search PDF Pages Without OCR](#10-demo-7--search-pdf-pages-without-ocr)
11. [Demo 8 — Matryoshka: Smaller Vectors](#11-demo-8--matryoshka-smaller-vectors)
12. [Demo 9 — Load Only What You Need](#12-demo-9--load-only-what-you-need)
13. [Key Results](#13-key-results)
14. [Rules of Thumb](#14-rules-of-thumb)
15. [Project Structure](#15-project-structure)
16. [Limitations & Future Work](#16-limitations--future-work)
17. [References](#17-references)

---

## 1. What This Project Does

A single Jupyter notebook that runs **9 real multimodal retrieval tasks** against public datasets using [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2) — an open-source (Apache 2.0), 270M-parameter embedding model from Google that maps **text, images, audio, and video** into the **same 768-dimensional space**.

Concretely the notebook answers these questions:

- "Can I find a photo with a German sentence?" — yes.
- "Can I find a photo with a sound clip?" — yes.
- "Can I search 1,000 photos by text in one round?" — yes, ~86% Recall@1.
- "Can I find a moment in a video with a text description?" — yes, with timestamps.
- "Can I search a PDF without OCR?" — yes, the vision encoder reads the page.
- "Can I shrink the index 6× by truncating the vectors?" — yes, with measured recall cost.

Everything runs end-to-end on a free Google Colab T4 GPU in about 15 minutes.

---

## 2. Quick Start

### Prerequisites

- Python 3.10+
- A GPU with at least 8 GB VRAM (a free Colab T4 works)
- ~3 GB of disk for the model + ~230 MB of demo data

### Run it

```bash
# Clone
cd embeddinggemma2multimodal

# (Optional) create a venv
uv venv && source .venv/bin/activate

# Install — see the comment in cell 2 for the gotchas
uv pip install -U "sentence-transformers[image,video]>=6.1.0" transformers torchcodec gTTS pymupdf

# Launch Jupyter
jupyter lab EmbeddingGemma2_Multimodal_Retrieval.ipynb
```

In Colab:

1. Open the notebook.
2. **Runtime → Change runtime type → T4 GPU.**
3. Run all cells (`Runtime → Run all`).
4. Wait ~15 minutes (most time goes to the 1,000-photo index in demo 2).

> **Install gotcha** (documented in the notebook): the `[audio]` extra pulls in `transformers[audio]` → `pyctcdecode`, which pins `numpy<2`. On Colab Python 3.13 there's no prebuilt NumPy 1.x, so pip compiles from source and downgrades NumPy. Audio works without that extra because Colab ships `librosa` + `soundfile` already.

### What's in the box

| Demo | Real-data signal | Visible output |
|---|---|---|
| 1 | One space for every modality | 3×6×6 similarity heatmaps |
| 2 | Photo search | Recall@1 = **86.4%**, Recall@5 = **97.5%** on Flickr 1k |
| 3 | Multilingual + photo-as-query | DE/ES/photo → photos |
| 4 | Voice search | audio (no transcript) → photos |
| 5 | Sound search | text → 50 ESC-50 clips, **24%** top-1 / **56%** top-5 |
| 6 | Video moments | text → timestamps in *Big Buck Bunny* |
| 7 | PDF without OCR | text → pages of *Attention Is All You Need* |
| 8 | Matryoshka | recall at 768/512/256/128 dims |
| 9 | Modular load | text-only 271M model vs full model, identical embeddings |

---

## 3. The 9 Demos

```
┌──────────────────────────────────────────────────────────────────────┐
│                   EmbeddingGemma 2 — one 768-D space                 │
│                                                                      │
│   ┌──────────────┐                                                   │
│   │   text query │ ─┐                                                │
│   │  ("a dog…")  │  │                                                │
│   └──────────────┘  │                                                │
│   ┌──────────────┐  │   cosine                                       │
│   │ audio query  │ ─┤   similarity     ┌─────────────────────────┐   │
│   │  (voice .wav)│  │   in one         │        index             │  │
│   └──────────────┘  │   shared         │  ┌──────────────────┐    │  │
│   ┌──────────────┐  │   vector        │  │ 1,000 photos    │    │  │
│   │ image query  │ ─┤   space  ─────► │  │ 50 sounds       │    │  │
│   │  (.jpg)      │  │                 │  │ 60 video chunks │    │  │
│   └──────────────┘  │                 │  │ 15 PDF pages    │    │  │
│   ┌──────────────┐  │                 │  └──────────────────┘    │  │
│   │ video query  │ ─┘                 └─────────────────────────┘   │
│   │  (.mp4)      │                                                    │
│   └──────────────┘                                                    │
└──────────────────────────────────────────────────────────────────────┘
```

The 9 demos walk left-to-right along this diagram — first proving the cross-modal alignment (demo 1), then exercising one query type against one real dataset at a time (demos 2–7), then pushing on the index side with vector compression (demo 8) and the model side with modular loading (demo 9).

---

## 4. Demo 1 — One Space for Every Modality

**Setup.** Six concepts (dog, crying baby, ocean waves, church, train, fire), each represented as a **sentence**, a **photo**, and a **sound clip**. If the three encoders share one vector space, matching pairs should score highest, showing up as a bright diagonal in the similarity heatmap.

**Why it matters.** The third heatmap — photo ↔ sound directly, with no text anywhere in between — is the hardest test of cross-modal alignment. Most older stacks (CLIP-style) cannot do this at all.

**Key code:**

```python
concepts = {'a dog': [...], 'a crying baby': [...], ...}

text_emb  = model.encode(
    [f"a photo or sound of {c}" for c in names], prompt_name="SearchQuery"
)
photo_emb = model.encode([{"image": str(...)} for c in names])        # no prompt for media
sound_emb = model.encode([{"audio":  str(...)} for c in names])

s = model.similarity(text_emb, photo_emb).float().numpy()             # text → photo
```

**Takeaway.**

- text → photo: clean diagonal, every sentence finds its photo.
- text → sound: most right, non-speech sounds are the weak spot (reappears in demo 5).
- photo → sound: hardest, noisiest. Non-speech sound is the model's soft side.

> The scores themselves are compressed in a narrow band. **Cosine similarity from an embedding model is only meaningful relative to the other scores** in the batch. Don't threshold at "0.8 = match" without calibrating on your own data.

---

## 5. Demo 2 — Search 1,000 Photos with Text

**Setup.** Embed every Flickr-1k photo once (that's the index). Each query is one text embedding + one matrix multiply. Slow part: ~8 minutes on a T4.

**Key code:**

```python
photo_index = model.encode(
    [{"image": p} for p in photo_paths], batch_size=16, show_progress_bar=True
)

for q in ["a dog catching a frisbee", "people dancing at a wedding",
          "a man fixing a bicycle", "kids playing in the snow"]:
    hits = top_k(model.encode(q, prompt_name="SearchQuery"), photo_index, k=4)
```

**Measured retrieval quality on Flickr 1k captions:**

| Metric | Value |
|---|---|
| Recall@1 (5,000 captions) | **0.8636** |
| Recall@5 (5,000 captions) | **0.9750** |

> Recall@1 = "is the right photo the top hit?" Recall@5 = "is it in the top 5?"

---

## 6. Demo 3 — Any Language, or a Photo as the Query

**Setup.** The text encoder handles 100+ languages out of the box, and the query doesn't have to be text at all.

**Examples that just work:**

```python
# German
hits = top_k(model.encode("ein Hund, der im Schnee rennt",
                           prompt_name="SearchQuery"), photo_index, k=4)

# Spanish
hits = top_k(model.encode("una mujer tocando la guitarra",
                           prompt_name="SearchQuery"), photo_index, k=4)

# Image as query (find similar photos)
hits = top_k(photo_index[319:320], photo_index, k=6)[1:]   # skip self
```

**Takeaway.** The model has no separate multilingual encoder — it's a single shared space. Same code, same `SearchQuery` prompt, different language.

---

## 7. Demo 4 — Ask with Your Voice

**Setup.** Synthesize spoken questions with `gTTS`, load them at 16 kHz mono, embed the **audio directly** — no speech-to-text, no transcript. The notebook uses `gTTS` so it runs anywhere, but any 16 kHz mono recording works, including a phone voice memo.

**Key code:**

```python
def speak(text, path):
    mp3 = path.with_suffix(".mp3"); gTTS(text).save(mp3)
    wav, _ = librosa.load(mp3, sr=16000, mono=True)    # model expects 16 kHz mono
    sf.write(path, wav, 16000); return str(path)

hits = top_k(model.encode({"audio": speak("a dog jumping into a lake", ...)}),
            photo_index, k=4)
```

**Takeaway.** The spoken query never becomes text, yet lands on almost the same photos as the typed sentence. Audio → photo works because **speech is the audio encoder's strongest domain** — voice search, meeting recordings, voice memos are the natural fit (see demo 5 for non-speech limits).

---

## 8. Demo 5 — Search Sounds with Text

**Setup.** Embed 50 environmental sound classes from ESC-50 (one clip per class — dog, rain, chainsaw, church bells, etc.). For every class name, does its own clip come back first?

**Key code:**

```python
classes = list(esc_first)
sound_index = model.encode(
    [{"audio": str(DATA / "esc50" / esc_first[c])} for c in classes], batch_size=8
)
label_emb = model.encode(
    [f"the sound of {c.replace('_', ' ')}" for c in classes], prompt_name="SearchQuery"
)
ranks = model.similarity(label_emb, sound_index).argsort(dim=1, descending=True).numpy()
```

**Measured zero-shot audio classification (chance = 2% / 10%):**

| Metric | Value |
|---|---|
| Top-1 accuracy | **24%** |
| Top-5 accuracy | **56%** |

**Takeaway.** Far above chance, well below the photo numbers. **General sound events are the weak spot.** If sound-event search matters to you, test on your own clips before relying on it. The 12 hits it gets right (`chirping_birds`, `vacuum_cleaner`, `clapping`, `church_bells`, `wind`, `crackling_fire`, `insects`, `laughing`, `hen`, `breathing`, `toilet_flush`, `clock_tick`) are the prototypical, well-separated ones.

---

## 9. Demo 6 — Video Moments Finder

**Setup.** Cut the first 5 minutes of *Big Buck Bunny* into 5-second chunks and embed each as video (1 fps sampling = 5 frames per chunk, 140 tokens per frame). A text query returns timestamps.

**Key code:**

```python
CHUNK = 5  # seconds
starts = list(range(0, 300, CHUNK))  # 60 chunks
chunk_paths = [...]                  # ffmpeg-cut chunks
video_index = model.encode(
    [{"video": p} for p in chunk_paths], batch_size=4, show_progress_bar=True
)

for q in ['a bird stretching its wings on a branch', 'a dark hole in the ground', ...]:
    hits = top_k(model.encode(q, prompt_name="SearchQuery"), video_index, k=3)
    # returns (chunk_index, score) → frame_at(starts[i] + 2.5)
```

**Takeaway.** The score ranks timestamps; the notebook renders the frame at the midpoint of each top-k chunk for visual confirmation.

---

## 10. Demo 7 — Search PDF Pages Without OCR

**Setup.** Render every page of *Attention Is All You Need* (the arXiv PDF) to an image with PyMuPDF and embed the **picture of the page**. The vision encoder reads layout, figures, and tables directly — no text extraction.

**Key code:**

```python
import pymupdf
pdf = pymupdf.open(DATA / "attention.pdf")
page_paths = []
for n, page in enumerate(pdf, start=1):
    out = DATA / f"page_{n:02d}.png"; page.get_pixmap(dpi=110).save(out)
    page_paths.append(str(out))

image_processor = model[0].processor.image_processor
page_index = {}
for budget in (280, 1120):                       # tokens per image
    image_processor.max_soft_tokens = budget
    page_index[budget] = model.encode(
        [{"image": p} for p in page_paths], batch_size=2
    )
```

**Takeaway.** On this paper the default 280-token budget already finds the right pages; bumping to 1,120 tokens cost ~9× longer without changing much. **Raise the budget when you need fine detail inside a page** (small numbers, dense tables) — not for finding which page a figure is on.

---

## 11. Demo 8 — Matryoshka: Smaller Vectors

**Setup.** EmbeddingGemma 2 is trained so the **first** 512, 256, or 128 numbers of each vector still work as an embedding on their own. Truncate → smaller index (768 → 128 is 6× less storage).

**Key code:**

```python
def truncate(e, d):
    e = np.asarray(e)[:, :d]
    return e / np.linalg.norm(e, axis=1, keepdims=True)    # re-normalize or cosine drifts

for d in (768, 512, 256, 128):
    r   = recall_at(truncate(cap_emb, d), truncate(photo_index, d), cap_owner)
    acc = (model.similarity(truncate(label_emb, d), truncate(sound_index, d))
           .argmax(1).numpy() == np.arange(len(classes))).mean()
    print(f"{d:>4} dims | {768/d:>3.1f}x smaller | photo R@1 {r['R@1']:.3f}  R@5 {r['R@5']:.3f} | sound top-1 {acc:.0%}")
```

**Measured recall vs vector size:**

| Dims | Index size | Photo R@1 | Photo R@5 | Sound top-1 |
|---|---|---|---|---|
| 768 | 1.0× | 0.864 | 0.975 | 24% |
| 512 | 1.5× smaller | 0.861 | 0.972 | 20% |
| 256 | 3.0× smaller | 0.842 | 0.967 | 22% |
| 128 | 6.0× smaller | 0.752 | 0.923 | 16% |

**Takeaway.** Text barely notices until 128. Multimodal drops faster at 128. In production:

```python
model.encode(query, prompt_name="SearchQuery",
             truncate_dim=256, normalize_embeddings=True)
```

Two rules: **re-normalize after truncating** (`normalize_embeddings=True` does it) and **queries and index must use the same size**.

---

## 12. Demo 9 — Load Only What You Need

**Setup.** The vision (170M) and audio (300M) encoders are *optional* modules on a 270M text backbone. All three share one vector space, so a **text-only model** can search the photo index built by the full model.

**Key code:**

```python
text_only = SentenceTransformer(
    MODEL_ID, device=device, model_kwargs={"torch_dtype": dtype},
    config_kwargs={"vision_config": None, "audio_config": None},
)
print(f"text-only model: {sum(p.numel() for p in text_only.parameters())/1e6:.0f}M parameters")

cap_emb_small = text_only.encode(all_caps, prompt_name="SearchQuery", batch_size=64)
print("full model queries     :", recall_at(cap_emb,     photo_index, cap_owner))
print("text-only model queries:", recall_at(cap_emb_small, photo_index, cap_owner))
print("max |difference|       :", float(np.abs(cap_emb_small - cap_emb).max()))
```

**Measured:**

| | Params | R@1 | R@5 |
|---|---|---|---|
| Full model queries | 308M (text 270M + vision 170M + audio 300M) | 0.8636 | 0.975 |
| Text-only model queries | **271M** | 0.8636 | 0.975 |
| Max │embedding diff│ between the two | — | **0.0** |

**Takeaway.** Photos indexed once on a server, queried from a phone with a 271M text-only model — and the embeddings are *bit-identical*. The modular architecture isn't marketing, it's a deployment contract.

---

## 13. Key Results

| Benchmark | Result | Baseline |
|---|---|---|
| Flickr 1k caption → photo R@1 | **0.864** | chance ≈ 0.001 |
| Flickr 1k caption → photo R@5 | **0.975** | chance ≈ 0.005 |
| ESC-50 zero-shot audio top-1 | **24%** | chance = 2% |
| ESC-50 zero-shot audio top-5 | **56%** | chance = 10% |
| Matryoshka 128d photo R@1 | 0.752 (vs 0.864 @ 768d) | — |
| Cross-modal similarity | bright diagonal text↔photo, noisier photo↔sound | — |
| Text-only model vs full | bit-identical embeddings | — |

---

## 14. Rules of Thumb

### Do

- **Use a task prompt on text:** `prompt_name="SearchQuery"` for queries, `"Document"` for passages. Use `"CodeRetrieval"`, `"QuestionAnswering"`, `"FactChecking"`, `"Classification"`, `"Clustering"`, `"SentenceSimilarity"` when they fit.
- **Pass media with no prompt:** `{"image": ...}`, `{"audio": ...}`, `{"video": ...}`.
- **For titled documents, write the prefix yourself:** `f"title: {title} | text: {body}"` (the `Document` prompt always says `title: none`).
- **Audio = 16 kHz mono.** Video = 1 fps by default.
- **Mix modalities** with placeholders: `{"text": "Trail shoe <|image|> grip <|video|>", "image": ..., "video": ...}` gives one vector for the whole listing.
- **Rank by similarity.** Calibrate any threshold on your own data.

### Avoid

- **`torch.float16`.** Use `bfloat16` (Ampere+) or `float32`. fp16 gives NaN or quietly wrong vectors.
- **Truncating without re-normalizing**, or comparing vectors of different sizes.
- **128 dims** for image / video / audio search unless you've measured it on your own data (demo 8).
- **Forgetting the token budget:** one input holds 8,192 tokens ≈ 29 images, ≈ 58 video frames, or ≈ 5.5 minutes of audio.

---

## 15. Project Structure

```
embeddinggemma2multimodal/
├── README.md                                       # this file
└── EmbeddingGemma2_Multimodal_Retrieval.ipynb      # the entire demo, end-to-end
```

The notebook is self-contained: it installs its own deps, downloads ~230 MB of public data, builds the indexes, and produces every plot and result in this README.

**Datasets:**

| Dataset | Used by | License |
|---|---|---|
| Flickr 1k test split | demos 1, 2, 3, 8 | [HuggingFace dataset card](https://huggingface.co/datasets/nlphuji/flickr_1k_test_image_text_retrieval) |
| ESC-50 | demos 1, 5 | CC BY-NC 3.0 |
| Big Buck Bunny (first 5 min) | demo 6 | CC BY 3.0 |
| *Attention Is All You Need* (arXiv) | demo 7 | arXiv non-exclusive license |

**Output of each cell** (when run) is inline: matplotlib figures, similarity heatmaps, top-k image rows, top-1 audio playback, top-k timestamps.

---

## 16. Limitations & Future Work

### Known limitations

- **English captions dominate on segmentation.** The text side is multilingual, but the caption distributions on Flickr 1k are skewed.
- **Audio weak spot is non-speech.** Speech (demo 4) is strong; environmental sounds (demo 5) are weaker.
- **GPU memory.** A T4 with 16 GB works; smaller cards may OOM on the 1,000-photo batch.
- **Video sampler is fixed** at 1 fps / 140 tokens per frame. Fine-grained retrieval (sub-second) needs chunking logic not shown.
- **PDF demo is small** (15 pages). Behavior on thousand-page documents not measured here.
- **No streaming.** All 1,000 photos fit in RAM; production usage of larger indexes would need FAISS / ScaNN.

### Future work

- FAISS index + real retrieval latency numbers (10K, 100K, 1M images).
- Late-interaction / ColBERT-style reranking for fine-grained PDF page retrieval.
- Audio chunking for sound-event search at >5 s clips.
- Quantization (int8 / binary) on top of Matryoshka truncation.
- Frame-level (vs clip-level) video retrieval.

---

## 17. References

### Model & docs

- [EmbeddingGemma 2 model card](https://huggingface.co/google/embeddinggemma-2) — full architecture, training, benchmarks
- [Developer guide](https://developers.googleblog.com/embeddinggemma-2-the-developer-guide/) — Google AI Developers blog
- [On-device guide (LiteRT, MediaPipe)](https://developers.googleblog.com/google-ai-edge-with-embeddinggemma-2/) — mobile deployment

### Libraries

- [sentence-transformers](https://www.sbert.net/) — wrapper used throughout
- [torchcodec](https://github.com/pytorch/torchcodec) — video decoding
- [PyMuPDF](https://pymupdf.readthedocs.io/) — PDF → image rendering
- [librosa](https://librosa.org/) + [soundfile](https://github.com/bastibe/python-soundfile) — audio loading
- [gTTS](https://pypi.org/project/gTTS/) — demo TTS for spoken queries
- [HuggingFace Hub](https://huggingface.co/docs/hub/) — dataset downloads

### Concepts

- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — the truncation trick in demo 8
- [CLIP](https://arxiv.org/abs/2103.00020) — the older image↔text alignment baseline
- [ImageBind](https://arxiv.org/abs/2305.05665) — Meta's earlier six-modality alignment model

---

## License

Apache 2.0 (matches the upstream model and notebook). Demo datasets retain their own licenses (see Project Structure).