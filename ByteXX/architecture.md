# SatQuery AI — Complete Architecture Flow

## System Overview

```mermaid
flowchart TD
    subgraph USER["👤 User (Browser)"]
        A["Upload Satellite Image\n(PNG/JPEG)"]
        B["Type a Question\ne.g. 'How many buildings?'"]
        C["Click Submit"]
        Z["See Answer on Screen\ne.g. '3 buildings detected'"]
    end

    subgraph FRONTEND["🖥️ Frontend (Someone else building)"]
        F1["Receive image + question"]
        F2["Upload image to\nFirebase Storage"]
        F3["Get image download URL\n(public HTTPS link)"]
        F4["Write to Realtime DB\n/queries/{id}\n• image_url\n• question\n• task = vqa\n• status = pending"]
        F5["Listen to\n/results/{id}\n(real-time)"]
        F6["Display answer\nto user"]
    end

    subgraph FIREBASE["🔥 Firebase (Google Cloud)"]
        FB_STORAGE["Firebase Storage\nstores the image file\nsatellite-efa0a.firebasestorage.app"]
        FB_QUERIES["/queries/{id}\nstatus: pending ➜ processing ➜ done"]
        FB_RESULTS["/results/{id}\nanswer, confidence, task"]
    end

    subgraph BACKEND["⚙️ FastAPI Backend (Your RTX 3050 PC)"]
        BE1["Firebase Listener\npolls DB every 2 sec"]
        BE2["Detects status = pending\nmarks it processing"]
        BE3["Downloads image\nfrom Storage URL"]
        BE4["Loads image into\nPIL / RAM"]
        BE5["Runs inference on\nFine-tuned Qwen2.5-VL-3B\n(4-bit, ~4GB VRAM)"]
        BE6["Gets answer + confidence\ne.g. answer='3', conf=0.87"]
        BE7["Writes to\n/results/{id}\nmarks query done"]
    end

    subgraph MODEL["🤖 Fine-tuned Model (RTX 3050)"]
        M1["Qwen2.5-VL-3B-Instruct\nBase model"]
        M2["+ LoRA Adapter\n(satquery-ai-vqa-lora)\ntrained on VRSBench 85K VQA samples"]
        M3["Input: image + question\nOutput: text answer"]
    end

    %% User flow
    A --> F1
    B --> F1
    C --> F1
    F1 --> F2
    F2 --> FB_STORAGE
    FB_STORAGE --> F3
    F3 --> F4
    F4 --> FB_QUERIES

    %% Backend picks up
    FB_QUERIES --> BE1
    BE1 --> BE2
    BE2 --> FB_QUERIES
    BE2 --> BE3
    BE3 --> FB_STORAGE
    FB_STORAGE --> BE3
    BE3 --> BE4
    BE4 --> BE5

    %% Model inference
    BE5 --> M1
    M1 --> M2
    M2 --> M3
    M3 --> BE6

    %% Write result back
    BE6 --> BE7
    BE7 --> FB_RESULTS
    BE7 --> FB_QUERIES

    %% Frontend gets result
    FB_RESULTS --> F5
    F5 --> F6
    F6 --> Z

    %% Styling
    style USER fill:#1a1a2e,stroke:#e94560,color:#fff
    style FRONTEND fill:#16213e,stroke:#0f3460,color:#fff
    style FIREBASE fill:#ff6d00,stroke:#ff8f00,color:#fff
    style BACKEND fill:#0d2137,stroke:#00b4d8,color:#fff
    style MODEL fill:#1b4332,stroke:#52b788,color:#fff
```

---

## Step-by-Step Flow (Plain English)

| Step | Who | What Happens |
|------|-----|-------------|
| 1 | **User** | Opens frontend, selects a satellite image and types a question |
| 2 | **Frontend** | Uploads the image to **Firebase Storage** (gets a download URL) |
| 3 | **Frontend** | Writes to **Realtime DB** `/queries/{id}` with `status: pending` |
| 4 | **Backend** | Firebase Listener detects the pending query (checks every 2 sec) |
| 5 | **Backend** | Updates `status: processing` so frontend knows it started |
| 6 | **Backend** | Downloads the image from Firebase Storage URL into RAM |
| 7 | **Model** | Qwen2.5-VL-3B + your LoRA adapter runs VQA inference on the image |
| 8 | **Model** | Returns answer text + confidence score |
| 9 | **Backend** | Writes answer to `/results/{id}`, sets `status: done` |
| 10 | **Frontend** | Was listening to `/results/{id}` — instantly receives the answer |
| 11 | **User** | Sees the answer on screen |

---

## Timing Breakdown

```
User clicks Submit
       │
       ▼
  ~2-5 sec    Image uploads to Firebase Storage
       │
       ▼
  ~0-2 sec    Backend listener detects pending query
       │
       ▼
  ~1-2 sec    Image downloads from Storage to backend
       │
       ▼
  ~5-15 sec   Qwen2.5-VL-3B runs inference (RTX 3050, 4-bit)
       │
       ▼
  ~0 sec      Answer written to Firebase, frontend notified instantly
       │
       ▼
Total: ~10-25 seconds end-to-end
```

---

## Component Responsibilities

```
┌─────────────────────────────────────────────────────────┐
│  FRONTEND DEV'S JOB                                     │
│  • File picker UI                                       │
│  • Upload to Firebase Storage                           │
│  • Write to /queries/{id} with status=pending           │
│  • Show loading spinner (status: processing)            │
│  • Listen to /results/{id} and display answer           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  YOUR BACKEND (Already Built ✅)                        │
│  • FastAPI server on your RTX 3050 PC                   │
│  • firebase_listener.py polls DB every 2 sec            │
│  • Downloads image → runs Qwen2.5-VL-3B inference       │
│  • Writes answer back to Firebase                       │
│  • Also has REST API (/api/vqa, /api/caption, etc.)     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  FIREBASE (Already Set Up ✅)                           │
│  • Storage: holds the uploaded images                   │
│  • Realtime DB: message queue between frontend/backend  │
│  • Acts as the "glue" — no direct connection needed     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  YOUR FINE-TUNED MODEL (Training in progress)           │
│  • Base: Qwen2.5-VL-3B-Instruct                         │
│  • Adapter: LoRA r=8, trained on VRSBench 85K VQA       │
│  • Specialty: satellite/aerial imagery understanding    │
│  • Loaded once at backend startup, reused for all req.  │
└─────────────────────────────────────────────────────────┘
```
