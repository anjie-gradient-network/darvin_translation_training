Docker Training (Darvin dataset format)

Prerequisites
- Linux host with NVIDIA GPU(s) and drivers installed.
- Docker Engine 24+ and Docker Compose plugin.
- NVIDIA Container Toolkit installed (nvidia-container-toolkit) so Docker can see GPUs.

Dataset format (required)
- Only Darvin ZIP dataset is supported (no CSV):
  - ZIP root must contain `dataset_schema.json` and `annotations.jsonl`.
  - Optional `assets/` folder for local resources.
  - See backend splitter rules for details.

Inputs and paths
- Base model baked into image at `${BASE_MODEL_DIR}` (default `/opt/darvin/base_model`).
  - For local compose, you can still mount `./basemodel` to `/tmp/workspace/source_model` and override `BASE_MODEL_DIR`.
- Training outputs saved to `/tmp/workspace/target_model` (from `./output`) or `${DARVIN_OUTPUT_DIR}` when launched by backend.
- Provide training data via one of:
  1) `DARVIN_TRAIN_BUNDLE_PATH`: path to a training bundle zip (contains `dataset/training/*.zip`).
  2) `DARVIN_DATASET_DIR`: a directory containing dataset zips or an extracted bundle at `dataset/training/*.zip`.

Quick start
- Build and run `trainer` (compose file provided):
  - Put your dataset bundle at `./dataset/training-bundle.zip` (example)
  - Mount repo to `/tmp/workspace/train` (already in compose)
  - Set `DARVIN_TRAIN_BUNDLE_PATH=/tmp/workspace/train/dataset/training-bundle.zip`

What runs under the hood
- Base image: `hiyouga/llamafactory:latest` (includes `llamafactory-cli`).
- Command executed (inside container):
  - `cd /tmp/workspace/train/train-0 && bash train.sh`
  - `train.sh` calls `prepare_v2.py` which now loads Darvin datasets (ZIP), writes `train.yaml`, and then runs `llamafactory-cli train train.yaml`.

Environment variables
- `BASE_MODEL_DIR` (optional): path to base model (default `/opt/darvin/base_model`).
- `DARVIN_TRAIN_BUNDLE_PATH` (preferred): bundle zip file path inside container.
- `DARVIN_DATASET_DIR` (fallback): directory containing dataset zips or extracted bundle (`dataset/training/*.zip`).
- `TARGET_MODEL_DIR`: output dir (default `/tmp/workspace/target_model`).
- `DARVIN_UPLOAD_URL`: backend API to upload the trained model artifact (`/api/training/upload`). When set, the runner zips `${TARGET_MODEL_DIR}` into `trained_model.zip`, uploads it with job auth, and includes the returned `trained_model_cid` in the final `completed` callback.

Notes
- CSV input is not supported anymore. Provide Darvin ZIP datasets only.
- If GPU doesn’t support BF16, adjust the auto-generated `train.yaml` or `prepare_v2.py` defaults (e.g., use FP16).
- For repeated runs, clear or version the `./output` directory to avoid overwriting.
