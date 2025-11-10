# lunar-sam2-demo

Segment Anything 2 (SAM2) playground with:

- **FastAPI backend** that loads a SAM2 checkpoint, runs inference from point/box prompts, and stores JSON + PNG masks on disk.
- **Vite + React frontend** that lets you upload an image, place positive/negative clicks, trigger segmentation, and inspect/download the results.

> 🛠️ The repo does not ship SAM2 weights. Download them from the official release and drop them in `weights/`.

---

## Project layout

```
backend/
  app/main.py              # FastAPI service
  requirements.txt
frontend/
  src/components/Segmenter.tsx
  ...
outputs/                   # Generated samples (gitignored)
weights/                   # Place SAM2 configs/checkpoints here (gitignored)
```

---

## Backend setup

1. **Create a virtualenv** and install dependencies:

   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Download SAM2 config + checkpoint** (e.g., `sam2.1_hiera_tiny.yaml/.pt`) and place them under `weights/`.

3. **Launch the API**:

   ```bash
   export SAM2_CONFIG_PATH=../weights/sam2.1_hiera_tiny.yaml
   export SAM2_CHECKPOINT_PATH=../weights/sam2.1_hiera_tiny.pt
   uvicorn app.main:app --reload --port 8000
   ```

   Environment variables you can override:

   | Variable | Default | Description |
   | --- | --- | --- |
   | `SAM2_CONFIG_PATH` | `weights/sam2.1_hiera_tiny.yaml` | YAML config for SAM2 |
   | `SAM2_CHECKPOINT_PATH` | `weights/sam2.1_hiera_tiny.pt` | Model checkpoint |
   | `SAM2_DEVICE` | `cuda` if available else `cpu` | Torch device |
   | `SAM2_OUTPUT_DIR` | `outputs` | Where JSON + masks are persisted |
   | `SAM2_ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |

### API surface

- `POST /segment` — multipart form with `image` file + `payload` JSON (points, labels, boxes, etc.). Returns masks sorted by score and a base64 overlay preview. Saves artifacts to `outputs/<uuid>/`.
- `GET /segment/{id}` — fetches the previously stored JSON metadata.
- `GET /healthz` — simple readiness probe.

---

## Frontend setup

1. Install dependencies:

   ```bash
   cd frontend
   npm install
   cp .env.example .env   # adjust VITE_API_BASE if backend runs elsewhere
   ```

2. Start the Vite dev server:

   ```bash
   npm run dev -- --host 0.0.0.0 --port 5173
   ```

3. Visit `http://localhost:5173`, upload an image, left-click for foreground prompts, right-click for background prompts, then press **Segment**.

The preview pane shows the overlay returned by the backend, and the JSON block contains COCO-compatible `rle`, `bbox`, and `area` values that you can ingest for downstream training jobs.

---

## Notes & next steps

- The backend saves `meta.json`, `original.png`, and one `mask_<n>.png` per response under `outputs/<uuid>`. You can batch these into your labeling pipeline.
- If you need persistent history or collaboration, back the metadata with a database and expose list/search endpoints.
- Add auth + storage hardening before exposing the service on the public internet.
