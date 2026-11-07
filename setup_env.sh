#!/usr/bin/env bash
# =============================================================================
# Uni-GraphLLM environment — supports vicuna / Qwen3 / Qwen3.5 backbones.
#
# Versions below are PINNED to a combo verified with Vicuna-7B and Qwen3-8B
# (full Cora train + inference). Use the same pins for a matching env.
#
# WHY torch 2.7.1: Qwen3 forward SIGFPEs on torch < 2.6; it works on >= 2.6.
# WHY no PyG C++ extensions: pyg-lib / torch-scatter / torch-sparse aren't needed
#   (repo never imports them directly; PyG's pure-python core is enough) and they
#   fail to build/load on old glibc.
#
# Requirements:
#   - NVIDIA driver recent enough for the CUDA 12.8 runtime in the cu128 wheels
#     (driver >= 525, ideally >= 535). Check with `nvidia-smi`.
#   - A CUDA GPU (sm_70..sm_90+). If the driver is too old, install a torch 2.7.1
#     build matching that driver's CUDA (e.g. cu126).
#
# Usage:  bash setup_env.sh [env_name]   then   conda activate <env_name>
# =============================================================================
set -euo pipefail

ENV_NAME="${1:-uni-graphllm}"

echo ">>> [1/6] create conda env: $ENV_NAME (python 3.10)"
conda create -n "$ENV_NAME" python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo ">>> [2/6] PyTorch 2.7.1 (CUDA 12.8) — torch>=2.6 REQUIRED for Qwen3"
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

echo ">>> [3/6] torch_geometric CORE ONLY (no C++ extensions, not needed by this repo)"
pip install "torch_geometric==2.8.0.post1"

echo ">>> [4/6] HuggingFace stack (transformers 5.14.1 has Qwen3 & Qwen3.5)"
pip install "transformers==5.14.1" "tokenizers==0.22.2" "peft==0.18.1" "accelerate==1.14.0"

echo ">>> [5/6] data / ML / eval dependencies"
pip install "sentencepiece==0.2.2" protobuf \
            "pandas==2.3.3" "scikit-learn==1.7.2" "scipy==1.15.3" "numpy==2.2.6" \
            "sentence_transformers==5.6.1" "datasets==5.0.0" "rapidfuzz==3.14.5" \
            "pytorch_lightning==2.6.5" "lightning==2.6.5"

echo ">>> [6/6] optional: rdkit (molecular datasets chempcba/chemhiv) + wandb (logging)"
pip install "rdkit==2026.03.4" "wandb==0.28.1" || echo "   (optional — safe to ignore if this fails)"

echo
echo ">>> sanity check"
python - <<'PY'
import torch, transformers, torch_geometric
print("torch          :", torch.__version__, "| cuda:", torch.version.cuda, "| avail:", torch.cuda.is_available())
print("transformers   :", transformers.__version__)
print("torch_geometric:", torch_geometric.__version__)
from transformers import Qwen3ForCausalLM, AutoTokenizer, LlamaForCausalLM
print("Qwen3ForCausalLM import: OK")
try:
    from transformers import Qwen3_5ForConditionalGeneration
    print("Qwen3_5 import: OK")
except Exception:
    print("Qwen3_5 import: NOT AVAILABLE (Qwen3 & vicuna still work)")
assert torch.__version__ >= "2.6", "torch too old for Qwen3!"
print("\nENV READY.")
PY
echo ">>> DONE. Activate with:  conda activate $ENV_NAME"
