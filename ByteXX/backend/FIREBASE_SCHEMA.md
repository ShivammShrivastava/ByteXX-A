# SatQuery AI — Firebase Schema & API Reference
# For the frontend developer

## Firebase Project
- **Project ID:** `satellite-efa0a`
- **Realtime DB URL:** `https://satellite-efa0a-default-rtdb.firebaseio.com`
- **Storage Bucket:** `satellite-efa0a.firebasestorage.app`

---

## Flow: How It Works

```
[Frontend]                        [Firebase]                    [Backend]
    |                                  |                             |
    |-- Upload image ----------------> Storage                       |
    |<- Get download URL --------------|                             |
    |-- Write query -----------------> /queries/{id} (status=pending)|
    |                                  |<-- Listener detects --------|
    |                                  |    pending query            |
    |                                  |----- Backend processes ---->|
    |                                  |<-- Writes answer ----------|
    |                                  |    /results/{id}            |
    |<-- Listen for result ------------|                             |
    |   (onValue /results/{id})        |                             |
```

---

## Step 1: Upload Image to Firebase Storage

```javascript
import { getStorage, ref, uploadBytes, getDownloadURL } from "firebase/storage";

const storage = getStorage(app);

async function uploadImage(file) {
  const imageRef = ref(storage, `uploads/${Date.now()}_${file.name}`);
  await uploadBytes(imageRef, file);
  const downloadURL = await getDownloadURL(imageRef);
  return downloadURL;  // pass this as image_url in Step 2
}
```

---

## Step 2: Write Query to Realtime Database

```javascript
import { getDatabase, ref, push, serverTimestamp } from "firebase/database";

const db = getDatabase(app);

async function submitQuery(imageUrl, question, task = "vqa") {
  const queriesRef = ref(db, "queries");

  // Push a new query — backend will detect this automatically
  const newQuery = await push(queriesRef, {
    image_url: imageUrl,    // from Step 1
    question:  question,    // user's question string
    task:      task,        // "vqa" | "caption" | "refer"
    status:    "pending",   // backend will update to "processing" → "done"
    timestamp: Date.now(),
  });

  return newQuery.key;  // queryId — use this in Step 3
}
```

**`task` values:**
| Value | Description |
|-------|-------------|
| `"vqa"` | Visual Question Answering — ask any question about the image |
| `"caption"` | Generate a detailed description of the image |
| `"refer"` | Find an object — set `question` to the referring expression |

---

## Step 3: Listen for Result

```javascript
import { getDatabase, ref, onValue, off } from "firebase/database";

const db = getDatabase(app);

function listenForResult(queryId, onResult, onError) {
  const resultRef = ref(db, `results/${queryId}`);

  const unsubscribe = onValue(resultRef, (snapshot) => {
    const data = snapshot.val();
    if (!data) return;  // result not ready yet

    onResult({
      answer:      data.answer,       // string: model's answer
      confidence:  data.confidence,   // float: 0.0 - 1.0
      task:        data.task,
      processedAt: data.processed_at, // Unix ms
    });

    unsubscribe();  // stop listening after getting result
  });

  // Optional: also listen to query status for loading states
  const queryRef = ref(db, `queries/${queryId}/status`);
  onValue(queryRef, (snapshot) => {
    const status = snapshot.val();
    // "pending" → "processing" → "done" | "error"
    console.log("Query status:", status);
    if (status === "error") {
      onError("Model processing failed");
    }
  });
}
```

---

## Full Example (Combined)

```javascript
async function askQuestion(imageFile, question) {
  // 1. Upload image
  const imageUrl = await uploadImage(imageFile);

  // 2. Submit query
  const queryId = await submitQuery(imageUrl, question, "vqa");

  // 3. Wait for result
  return new Promise((resolve, reject) => {
    listenForResult(queryId, resolve, reject);
  });
}

// Usage:
const result = await askQuestion(myImageFile, "How many buildings are visible?");
console.log(result.answer);     // "3"
console.log(result.confidence); // 0.87
```

---

## Database Structure (Realtime DB)

```
satellite-efa0a-default-rtdb/
├── queries/
│   └── {queryId}/
│       ├── image_url:    "https://firebasestorage.../..."
│       ├── question:     "How many buildings?"
│       ├── task:         "vqa"
│       ├── status:       "pending" | "processing" | "done" | "error"
│       └── timestamp:    1725441600000
│
└── results/
    └── {queryId}/
        ├── answer:       "3"
        ├── confidence:   0.87
        ├── task:         "vqa"
        └── processed_at: 1725441605000
```

---

## Firebase Config (for frontend)

```javascript
const firebaseConfig = {
  apiKey:            "AIzaSyDJtjpV4DpD-Ev8BeRJDsfZV4k5U63dpW4",
  authDomain:        "satellite-efa0a.firebaseapp.com",
  projectId:         "satellite-efa0a",
  storageBucket:     "satellite-efa0a.firebasestorage.app",
  messagingSenderId: "504899672780",
  appId:             "1:504899672780:web:4f61a1b212930752cdc069",
  measurementId:     "G-TY956H2RJF",
  databaseURL:       "https://satellite-efa0a-default-rtdb.firebaseio.com",
};
```

---

## Firebase Security Rules

### Realtime Database Rules
Set at: Firebase Console → Realtime Database → Rules

```json
{
  "rules": {
    "queries": {
      ".read": true,
      ".write": true
    },
    "results": {
      ".read": true,
      ".write": false
    },
    "_health_check": {
      ".read": true,
      ".write": false
    },
    "_schema": {
      ".read": true,
      ".write": false
    }
  }
}
```

### Storage Rules
Set at: Firebase Console → Storage → Rules

```
rules_version = '2';
service firebase.storage {
  match /b/{bucket}/o {
    match /uploads/{allPaths=**} {
      allow read: if true;
      allow write: if request.resource.size < 10 * 1024 * 1024  // 10 MB max
                   && request.resource.contentType.matches('image/.*');
    }
  }
}
```

---

## Status Codes

| `status` value | Meaning |
|----------------|---------|
| `"pending"` | Query written, waiting for backend to pick up |
| `"processing"` | Backend is running inference (takes 5-15 sec) |
| `"done"` | Answer is ready in `/results/{queryId}` |
| `"error"` | Something went wrong — check `/queries/{queryId}/error` |

---

## Timing

- Image upload: ~2-5 seconds (depends on image size)
- Backend picks up query: within 2 seconds of writing
- Model inference: ~5-15 seconds (RTX 3050, 4-bit)
- **Total end-to-end: ~10-20 seconds**
