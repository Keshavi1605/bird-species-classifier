import os
from pathlib import Path
from typing import Dict, Tuple
from urllib.request import urlretrieve

import gradio as gr
import torch

from src.model_utils import get_transforms, load_checkpoint

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/bird_classifier.pt"))
MODEL_URL = os.getenv("MODEL_URL", "").strip()
DEFAULT_MIN_TOP1_CONF = float(os.getenv("MIN_TOP1_CONF", "0.70"))
DEFAULT_MIN_MARGIN = float(os.getenv("MIN_MARGIN", "0.20"))


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Predictor:
    def __init__(self) -> None:
        self.device = choose_device()
        self.model = None
        self.idx_to_class = None
        self.preprocess = None
        self.model_error = None
        self._load()

    def _load(self) -> None:
        if not MODEL_PATH.exists() and MODEL_URL:
            try:
                MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                print(f"Downloading model from MODEL_URL to {MODEL_PATH} ...")
                urlretrieve(MODEL_URL, MODEL_PATH)
            except Exception as exc:
                self.model_error = f"Model download failed from MODEL_URL: {exc}"
                return
        if not MODEL_PATH.exists():
            self.model_error = (
                f"Model not found at {MODEL_PATH}. Train first with `python train.py`, set MODEL_PATH, or set MODEL_URL."
            )
            return
        try:
            model, idx_to_class, image_size = load_checkpoint(MODEL_PATH, self.device)
            self.model = model
            self.idx_to_class = idx_to_class
            self.preprocess = get_transforms(image_size=image_size, train=False)
            self.model_error = None
        except Exception as exc:
            self.model_error = f"Failed to load model: {exc}"

    @torch.no_grad()
    def predict(self, image, min_top1_conf: float, min_margin: float) -> Tuple[Dict[str, float], str]:
        if image is None:
            raise gr.Error("Please capture or upload an image.")
        if self.model_error:
            raise gr.Error(self.model_error)

        x = self.preprocess(image).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1).cpu()[0]
        top_probs, top_indices = torch.topk(probs, k=min(5, probs.numel()))
        top1 = float(top_probs[0].item())
        top2 = float(top_probs[1].item()) if top_probs.numel() > 1 else 0.0
        margin = top1 - top2

        species_scores = {
            self.idx_to_class[idx.item()].replace("_", " "): float(prob.item())
            for prob, idx in zip(top_probs, top_indices)
        }
        if top1 < min_top1_conf or margin < min_margin:
            details = (
                "### Not a clear bird result\n"
                f"- Top confidence: `{top1:.1%}`\n"
                f"- Margin vs 2nd guess: `{margin:.1%}`\n"
                "- Try a clearer, closer bird image."
            )
            return {"NOT A BIRD / LOW CONFIDENCE": 1.0, **species_scores}, details

        top_name = next(iter(species_scores.keys()))
        details = (
            "### Bird detected\n"
            f"- Predicted species: `{top_name}`\n"
            f"- Confidence: `{top1:.1%}`\n"
            f"- Margin vs 2nd guess: `{margin:.1%}`"
        )
        return species_scores, details


predictor = Predictor()

THEME_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@500&display=swap');

:root {
  --bg-main: #09111d;
  --bg-card: #101a2b;
  --text-main: #ecf2ff;
  --text-muted: #9fb0ca;
  --accent: #1dd3b0;
  --accent-2: #f59f00;
}

body, .gradio-container {
  font-family: "Space Grotesk", "Trebuchet MS", sans-serif !important;
  color: var(--text-main);
  background:
    radial-gradient(1200px 500px at 10% -20%, rgba(29, 211, 176, 0.18), transparent 55%),
    radial-gradient(900px 450px at 95% 0%, rgba(245, 159, 0, 0.2), transparent 52%),
    linear-gradient(170deg, #050b14 0%, #0a1220 60%, #0b1322 100%);
}

.app-shell {
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 18px;
  padding: 16px;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.02));
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
}

.title {
  font-size: 2rem;
  margin-bottom: 0.25rem;
  letter-spacing: 0.2px;
}

.subtitle {
  color: var(--text-muted);
  margin-bottom: 0.6rem;
}

.mono {
  font-family: "IBM Plex Mono", Menlo, monospace;
  color: var(--text-muted);
}

button {
  transition: transform 120ms ease, box-shadow 120ms ease !important;
}

button:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.28) !important;
}
"""

with gr.Blocks(title="Bird Species Classifier") as demo:
    gr.Markdown(
        """
        <div class="title">Bird Species Classifier</div>
        <div class="subtitle">Upload or capture a bird photo. The app predicts top species and flags uncertain/non-bird images.</div>
        <div class="mono">Model: ResNet18 fine-tuned on your dataset</div>
        """,
        elem_classes=["app-shell"],
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=11):
            image_input = gr.Image(type="pil", sources=["upload", "webcam"], label="Bird Image")
            with gr.Row():
                predict_btn = gr.Button("Classify Bird", variant="primary")
                clear_btn = gr.Button("Clear", variant="secondary")

        with gr.Column(scale=9):
            pred_output = gr.Label(num_top_classes=5, label="Top Predictions")
            details_output = gr.Markdown("### Waiting for image")
            with gr.Accordion("Advanced Thresholds", open=False):
                min_conf = gr.Slider(
                    minimum=0.50,
                    maximum=0.95,
                    step=0.01,
                    value=DEFAULT_MIN_TOP1_CONF,
                    label="Minimum Top-1 Confidence",
                )
                min_margin = gr.Slider(
                    minimum=0.05,
                    maximum=0.40,
                    step=0.01,
                    value=DEFAULT_MIN_MARGIN,
                    label="Minimum Margin (Top1 - Top2)",
                )

    predict_btn.click(
        fn=predictor.predict,
        inputs=[image_input, min_conf, min_margin],
        outputs=[pred_output, details_output],
    )
    image_input.change(
        fn=predictor.predict,
        inputs=[image_input, min_conf, min_margin],
        outputs=[pred_output, details_output],
    )
    clear_btn.click(lambda: (None, None, "### Waiting for image"), outputs=[image_input, pred_output, details_output])


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, css=THEME_CSS)
