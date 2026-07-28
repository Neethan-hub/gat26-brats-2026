# GAT-26 clean-room Task-3 (BraTS-GoAT) release container — selected ResEnc-M.
# Linux/AMD64, CUDA <=13.0 (A10G 24 GB target). Zero runtime network: every dependency and every
# weight is baked at build time; the entrypoint never downloads or caches. Weights and dataset/plans
# JSON are supplied via the PRIVATE build context (git-ignored) — they are NOT in git.
#
# Build (on a host with the private context assembled under ./context):
#   docker build --platform=linux/amd64 -f configs/release/Dockerfile -t gat26-goat-m:CANDIDATE ./context
# Test locally exactly as the organizers will (read-only /input, flat /output, no network). Genuine-
# A10G validation MUST pass --gpus all and gate on the exact GPU name (aborts before inference if the
# single visible GPU is not NVIDIA A10G):
#   docker run --rm --gpus all --network none --runtime=nvidia \
#     --memory=48G --shm-size=16G \
#     -e GAT26_REQUIRE_GPU_NAME="NVIDIA A10G" \
#     -v /ABS/INPUT:/input:ro -v /ABS/OUTPUT:/output:rw gat26-goat-m:CANDIDATE

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    NNUNET_DISABLE_UPDATE_CHECK=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    GAT26_WEIGHTS_DIR=/opt/gat26/weights

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3-pip python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# --- dependencies (pinned; installed at BUILD time only) ---
# torch/cu128 wheels from the PyTorch index; the nnU-Net stack pinned from the committed lock.
COPY requirements.lock.txt /opt/gat26/requirements.lock.txt
RUN python3 -m pip install --no-cache-dir --break-system-packages \
        --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 \
 && python3 -m pip install --no-cache-dir --break-system-packages \
        -r /opt/gat26/requirements.lock.txt

# --- runner code (tracked) ---
COPY scripts/release_infer.py           /opt/gat26/scripts/release_infer.py
COPY scripts/g4_reconstruct_validate.py /opt/gat26/scripts/g4_reconstruct_validate.py
COPY scripts/g3_audit_labeled_archive.py /opt/gat26/scripts/g3_audit_labeled_archive.py

# --- baked, NON-tracked artifacts supplied by the private build context ---
#   context/weights/fold_{0..4}/checkpoint_final.pth   (five DISTINCT M checkpoints after G7;
#     preliminary candidate uses --allow-duplicate-proxy with a single M0 slot — no accuracy claim)
#   context/plans/nnUNetResEncUNetMPlans.json  context/plans/dataset.json
COPY weights/ /opt/gat26/weights/
COPY plans/   /opt/gat26/plans/

WORKDIR /opt/gat26
# read-only /input, flat /output, deterministic, explicit nonzero exit on any invalid input/output.
ENTRYPOINT ["python3", "/opt/gat26/scripts/release_infer.py", \
            "--input", "/input", "--output", "/output", \
            "--weights-dir", "/opt/gat26/weights", \
            "--plans-json", "/opt/gat26/plans/nnUNetResEncUNetMPlans.json", \
            "--dataset-json", "/opt/gat26/plans/dataset.json"]
