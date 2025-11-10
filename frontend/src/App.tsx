import { Segmenter } from "./components/Segmenter.tsx";

function App() {
  return (
    <main style={{ maxWidth: "1200px", margin: "0 auto", padding: "2rem" }}>
      <header style={{ marginBottom: "1.5rem" }}>
        <h1 style={{ marginBottom: "0.25rem" }}>Segment Anything 2 Demo</h1>
        <p style={{ color: "#475569" }}>
          Upload an image, drop positive (left click) and negative (right click) prompts, and save the resulting
          segmentation JSON + mask overlays.
        </p>
      </header>
      <Segmenter />
    </main>
  );
}

export default App;
