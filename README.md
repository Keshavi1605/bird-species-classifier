# Bird Species Classifier Web App

Train a bird classifier from your dataset and run a mobile-friendly web app with:
- camera/upload input
- top-5 species predictions
- non-bird / low-confidence rejection
- adjustable confidence thresholds in UI

## 1) Run locally (Python)

```bash
cd "/Users/kesh/Documents/New project"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Train:

```bash
python train.py --data-dir "/Users/kesh/Documents/Projects/Bird Species Detection/100-bird-species"
```

Start app:

```bash
python app.py
```

Open:
- laptop: `http://localhost:7860`
- phone on same Wi-Fi: `http://<your-laptop-ip>:7860`

## 2) Run locally (Docker)

```bash
cd "/Users/kesh/Documents/New project"
docker build -t bird-classifier-app .
docker run --rm -p 7860:7860 -v "$(pwd)/models:/app/models" bird-classifier-app
```

## 3) Deploy officially

## Option A: Hugging Face Spaces (recommended)

1. Create a new **Gradio** Space.
2. Push this code (without large dataset).
3. Provide model one of two ways:
   - commit `models/bird_classifier.pt` (if you want it in repo), or
   - set `MODEL_URL` in Space variables to a direct download link for your model.
4. Space auto-builds and gives a public URL.

## Option B: Render (Docker)

1. Push this repo to GitHub.
2. In Render, create service from repo using `render.yaml`.
3. Set `MODEL_URL` env var in Render (or include model in image/repo).
4. Deploy.

## 4) Push to GitHub

Your `gh` token is currently expired, so re-auth first:

```bash
gh auth login -h github.com
```

Then create repo + push:

```bash
cd "/Users/kesh/Documents/New project"
git checkout -b codex/bird-app-deploy
git add .
git commit -m "Build bird classifier web app with polished UI and deployment setup"
gh repo create bird-species-classifier --public --source=. --remote=origin --push
```

If repo already exists on GitHub, use:

```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin codex/bird-app-deploy
```

## Environment variables

- `MODEL_PATH` default: `models/bird_classifier.pt`
- `MODEL_URL` optional: direct URL to download model at startup
- `MIN_TOP1_CONF` default: `0.70`
- `MIN_MARGIN` default: `0.20`

## Project files

- `app.py` Gradio web app
- `train.py` model training script
- `src/model_utils.py` model/transforms/checkpoint helpers
- `Dockerfile` container build
- `render.yaml` Render blueprint
- `.github/workflows/ci.yml` syntax CI
