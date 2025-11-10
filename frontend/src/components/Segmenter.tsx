import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";

type PromptPoint = { x: number; y: number; label: 0 | 1 };

type MaskPayload = {
  score: number;
  bbox: number[];
  area: number;
  rle: { counts: string; size: number[] };
};

type SegmentResponse = {
  id: string;
  file_name: string;
  preview_overlay: string;
  masks: MaskPayload[];
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

const canvasStyle: React.CSSProperties = {
  width: "100%",
  border: "1px solid #cbd5f5",
  borderRadius: "0.5rem",
  cursor: "crosshair",
  backgroundSize: "contain",
  backgroundRepeat: "no-repeat",
  backgroundPosition: "top left"
};

export function Segmenter() {
  const [file, setFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string>("");
  const [imageSize, setImageSize] = useState<{ width: number; height: number }>({ width: 0, height: 0 });
  const [points, setPoints] = useState<PromptPoint[]>([]);
  const [isUploading, setUploading] = useState(false);
  const [response, setResponse] = useState<SegmentResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!file) return;
    const objectUrl = URL.createObjectURL(file);
    setImageUrl(objectUrl);
    const img = new Image();
    img.onload = () => {
      setImageSize({ width: img.width, height: img.height });
    };
    img.src = objectUrl;
    return () => {
      URL.revokeObjectURL(objectUrl);
    };
  }, [file]);

  const addPoint = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement, MouseEvent>, label: 0 | 1) => {
      if (!canvasRef.current) return;
      const { width, height } = imageSize;
      if (!width || !height) return;
      const rect = canvasRef.current.getBoundingClientRect();
      const scaleX = width / rect.width;
      const scaleY = height / rect.height;
      const x = (event.clientX - rect.left) * scaleX;
      const y = (event.clientY - rect.top) * scaleY;
      setPoints((prev) => [...prev, { x, y, label }]);
    },
    [imageSize]
  );

  const removePoint = useCallback((index: number) => {
    setPoints((prev) => prev.filter((_, idx) => idx !== index));
  }, []);

  const reset = () => {
    setPoints([]);
    setResponse(null);
    setError(null);
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const nextFile = event.target.files?.[0];
    if (!nextFile) return;
    setFile(nextFile);
    reset();
  };

  const submit = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const payload = {
        points: points.map(({ x, y }) => [x, y]),
        labels: points.map(({ label }) => label),
        multimask: true,
        top_k: 3
      };
      const formData = new FormData();
      formData.append("image", file);
      formData.append("payload", JSON.stringify(payload));
      const res = await fetch(`${API_BASE}/segment`, { method: "POST", body: formData });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || "Segmentation failed");
      }
      const data = (await res.json()) as SegmentResponse;
      setResponse(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setUploading(false);
    }
  };

  const pointList = useMemo(
    () =>
      points.map((point, index) => (
        <li key={`${point.x}-${point.y}-${index}`} className="point-row">
          <span className={clsx("pill", point.label ? "pill-success" : "pill-danger")}>
            {point.label ? "FG" : "BG"}
          </span>
          <span>{`(${point.x.toFixed(1)}, ${point.y.toFixed(1)})`}</span>
          <button type="button" onClick={() => removePoint(index)}>
            ×
          </button>
        </li>
      )),
    [points, removePoint]
  );

  return (
    <section className="stack">
      <label className="upload">
        <span>Select an image</span>
        <input type="file" accept="image/*" onChange={handleFileChange} />
      </label>

      {imageUrl && (
        <>
          <div className="canvas-wrapper">
            <canvas
              ref={canvasRef}
              width={imageSize.width}
              height={imageSize.height}
              style={{ ...canvasStyle, backgroundImage: `url(${imageUrl})` }}
              onClick={(event) => addPoint(event, 1)}
              onContextMenu={(event) => {
                event.preventDefault();
                addPoint(event, 0);
              }}
            />
            <p className="hint">Left click = foreground, right click = background</p>
          </div>
          <div className="actions">
            <button type="button" onClick={() => setPoints([])} disabled={!points.length}>
              Clear Points
            </button>
            <button type="button" onClick={submit} disabled={!points.length || isUploading}>
              {isUploading ? "Running..." : "Segment"}
            </button>
          </div>
          <ul className="point-list">{pointList}</ul>
        </>
      )}

      {error && <p className="error">{error}</p>}

      {response && (
        <div className="results">
          <div>
            <h3>Mask overlay</h3>
            <img src={response.preview_overlay} alt="Mask overlay" className="preview" />
          </div>
          <div className="json">
            <h3>JSON response</h3>
            <pre>{JSON.stringify(response, null, 2)}</pre>
          </div>
        </div>
      )}
    </section>
  );
}
