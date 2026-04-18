# Challenge Submission Guide

This guide explains how to package your trained model into a self-contained Docker image, export it to a `.tar` file, and submit it to the challenge server.

It is based on:
- `predict.py`, including `Model`, `preprocess`, `postprocess`, and `main`
- `model.py`
- `Dockerfile`
- `train.py`

---

## 1. What the server expects

Your container runs `predict.py` as entrypoint (see `Dockerfile`).

Inside the container, paths are fixed:

- Input images: `IMAGE_DIR = "/data"`
- Output predictions: `OUTPUT_DIR = "/output"`
- Model weights: `MODEL_PATH = "/app/model.pt"`

`predict.py` loads `Model` with strict weights loading.
So your `/app/model.pt` **must match** the architecture in `model.py`.

---

## 2. Put your best trained checkpoint in place

Training saves checkpoints from `train.py`, typically under:
`checkpoints/<experiment-id>/...`

Pick your best checkpoint and copy/rename it to:

- `Final assignment/model.pt`

Example from repo root:

```bash
cp "Final assignment/checkpoints/unet-training/best_model-epoch=....pt" "Final assignment/model.pt"
```

On Windows PowerShell:

```powershell
Copy-Item "Final assignment\checkpoints\unet-training\best_model-epoch=....pt" "Final assignment\model.pt"
```

---

## 3. Build the Docker image

From repo root:

```bash
docker build -t nncv-submission:latest -f "Final assignment/Dockerfile" "Final assignment"
```

This creates a self-contained image with:
- `predict.py`
- `model.py`
- `model.pt`

---

## 4. Test locally before exporting

Create local folders:
- `./local_data` with `.png` images
- `./local_output` for predictions

Run (Linux/macOS shell):

```bash
docker run --rm \
  -v "$(pwd)/local_data:/data" \
  -v "$(pwd)/local_output:/output" \
  nncv-submission:latest
```

Run (Windows PowerShell):

```powershell
docker run --rm `
  -v "${PWD}\local_data:/data" `
  -v "${PWD}\local_output:/output" `
  nncv-submission:latest
```

Expected behavior:
- It reads all `*.png` from `/data`
- It writes predicted masks to `/output` (same filenames)

---

## 5. Export image to `.tar` for submission

```bash
docker save -o nncv_submission.tar nncv-submission:latest
```

You then submit `nncv_submission.tar`.

---

## 6. Challenge server endpoints

These servers are only reachable from TU/e network or VPN.

1. **Baseline / Peak Performance**  
	http://131.155.126.249:5001/

2. **Robustness**  
	http://131.155.126.249:5002/

3. **Efficiency**  
	http://131.155.126.249:5003/

4. **Out-of-Distribution**  
	http://131.155.126.249:5004/

---

## 7. Recommended workflow per benchmark

- Keep one stable baseline image/tag.
- Create separate model variants per benchmark.
- Export each as a separate tar, for example:
  - `submission_baseline_peak.tar`
  - `submission_robustness.tar`
  - `submission_efficiency.tar`
  - `submission_ood.tar`

Example:

```bash
docker build -t nncv-submission:efficiency -f "Final assignment/Dockerfile" "Final assignment"
docker save -o submission_efficiency.tar nncv-submission:efficiency
```

---

## 8. Common failure checks

- `model.pt` missing from `Final assignment/` before build.
- `model.pt` incompatible with `Model` in `model.py`.
- Predictions not saved as single-channel class-index PNG masks.
- Input/output paths changed from `/data` and `/output`.
- Built from wrong Docker context (must include `predict.py`, `model.py`, `model.pt`).

---

Good practice: test the container locally end-to-end before every submission.
