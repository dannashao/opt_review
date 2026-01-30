set -euo pipefail

# ============================================================
# HADES GPU setup (Linux + Conda + TensorFlow + legacy .h5)
# - Uses system CUDA toolkit for XLA "libdevice" (toolkit DATA)
# - Uses pip NVIDIA wheels for CUDA runtime libs (shared objects)
# - Avoids conda CUDA toolkit conflicts (especially with RAPIDS/cuML)
# ============================================================

# ---------- User settings ----------
ENV_NAME="HADES"
PY_VER="3.10"

# Where Ubuntu/Debian installs CUDA toolkit data (libdevice)
CUDA_TOOLKIT_ROOT="/usr/lib/nvidia-cuda-toolkit"

# Root directory containing .h5 models to patch (edit as needed)
H5_ROOT="pyphenotyper/model_refrence"
# -----------------------------------

echo "==> [1/8] Check NVIDIA driver"
# Why: driver must be installed and working (CUDA runtime depends on it)
command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: nvidia-smi not found. Install NVIDIA driver first."; exit 1; }
nvidia-smi || { echo "ERROR: nvidia-smi failed. Fix driver installation."; exit 1; }

echo "==> [2/8] Ensure system CUDA toolkit is installed (for libdevice/ptx toolchain)"
# Why: TensorFlow/XLA needs CUDA *toolkit data* (libdevice) for compiling certain GPU kernels.
# Pip provides runtime libs, but not the full toolkit layout XLA expects.
if [ ! -f "$CUDA_TOOLKIT_ROOT/libdevice/libdevice.10.bc" ]; then
  echo "CUDA libdevice not found at $CUDA_TOOLKIT_ROOT/libdevice/libdevice.10.bc"
  echo "Installing nvidia-cuda-toolkit (Ubuntu/Debian)..."
  sudo apt update
  sudo apt install -y nvidia-cuda-toolkit
fi

echo "==> [3/8] Create symlink to match XLA's expected CUDA layout"
# Why: XLA expects <CUDA_ROOT>/nvvm/libdevice/libdevice.10.bc.
# Ubuntu puts libdevice in <CUDA_ROOT>/libdevice/libdevice.10.bc, so we symlink.
sudo mkdir -p "$CUDA_TOOLKIT_ROOT/nvvm/libdevice"
sudo ln -sf "$CUDA_TOOLKIT_ROOT/libdevice/libdevice.10.bc" \
  "$CUDA_TOOLKIT_ROOT/nvvm/libdevice/libdevice.10.bc"

# Optional sanity check
test -f "$CUDA_TOOLKIT_ROOT/nvvm/libdevice/libdevice.10.bc" || { echo "ERROR: libdevice symlink missing"; exit 1; }

echo "==> [4/8] Create fresh conda env"
# Why: clean environment avoids Windows pins and solver conflicts.
# Note: requires conda already installed.
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda env '$ENV_NAME' already exists. Skipping create."
else
  conda create -n "$ENV_NAME" "python=$PY_VER" pip -y
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "==> [5/8] Install TensorFlow + NVIDIA CUDA runtime libs via pip"
# Why: pip TF + NVIDIA wheels provide CUDA runtime libraries in site-packages.
# This avoids conda CUDA toolkit installs, which often conflict with RAPIDS/cuML variants.
pip install -U "tensorflow==2.20.*"
pip install -U \
  nvidia-cuda-runtime-cu12 \
  nvidia-cublas-cu12 \
  nvidia-cudnn-cu12 \
  nvidia-cufft-cu12 \
  nvidia-curand-cu12 \
  nvidia-cusolver-cu12 \
  nvidia-cusparse-cu12 \
  nvidia-nccl-cu12

echo "==> [6/8] Add activation hooks: LD_LIBRARY_PATH + XLA_FLAGS"
# Why (LD_LIBRARY_PATH): NVIDIA pip wheels install .so files under site-packages/nvidia/*/lib,
# which is not on the default dynamic linker search path.
# Why (XLA_FLAGS): Point XLA to CUDA toolkit root so it can find nvvm/libdevice.
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/nvidia-libs.sh" <<'EOF'
# Add all NVIDIA pip wheel library dirs to LD_LIBRARY_PATH
# so TensorFlow can dlopen libcudart/libcublas/libcudnn/etc.
export LD_LIBRARY_PATH="$(python - <<'PY'
import site, glob, os
paths=set()
for sp in site.getsitepackages():
    for p in glob.glob(os.path.join(sp, "nvidia/*/lib")):
        paths.add(p)
print(":".join(sorted(paths)))
PY
):$LD_LIBRARY_PATH"
EOF

cat > "$CONDA_PREFIX/etc/conda/activate.d/xla_cuda.sh" <<EOF
# Point XLA to CUDA toolkit data dir (libdevice) and allow driver fallback if ptxas isn't available.
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_TOOLKIT_ROOT --xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found"
EOF

echo "==> [7/8] OPTIONAL: install sitecustomize to load legacy .h5 without compile"
# Why: legacy .h5 training configs often fail to deserialize metrics/optimizers in Keras 3 / TF 2.20.
# For inference, compile is unnecessary, so we default compile=False globally.
SITE_PY="$(python - <<'PY'
import sys, os
print(os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "sitecustomize.py"))
PY
)"
cat > "$SITE_PY" <<'EOF'
# Auto-disable compile on load_model for legacy .h5 files
try:
    import tensorflow as tf
    _real = tf.keras.models.load_model
    def _load_model(*args, **kwargs):
        kwargs.setdefault("compile", False)
        return _real(*args, **kwargs)
    tf.keras.models.load_model = _load_model
except Exception:
    pass

# If code uses standalone keras instead of tf.keras
try:
    import keras
    _real_k = keras.models.load_model
    def _load_model_k(*args, **kwargs):
        kwargs.setdefault("compile", False)
        return _real_k(*args, **kwargs)
    keras.models.load_model = _load_model_k
except Exception:
    pass
EOF

echo "==> [8/8] OPTIONAL: patch legacy .h5 models in place (remove 'groups' key)"
# Why: older models may store Conv2DTranspose config containing "groups": 1.
# Modern Keras rejects this for Conv2DTranspose, even though groups=1 is a no-op.
# This patch edits model_config JSON inside each .h5, leaving weights untouched.
if [ -d "$H5_ROOT" ]; then
  python - <<EOF
import os, json
import h5py

ROOT = "${H5_ROOT}"

def strip_groups(obj):
    if isinstance(obj, dict):
        obj.pop("groups", None)
        for v in obj.values():
            strip_groups(v)
    elif isinstance(obj, list):
        for v in obj:
            strip_groups(v)

patched = 0
for root, _, files in os.walk(ROOT):
    for fn in files:
        if not fn.endswith(".h5"):
            continue
        path = os.path.join(root, fn)
        with h5py.File(path, "r+") as f:
            mc = f.attrs.get("model_config")
            if mc is None:
                continue
            if isinstance(mc, bytes):
                mc = mc.decode("utf-8")
            if '"groups"' not in mc:
                continue
            cfg = json.loads(mc)
            strip_groups(cfg)
            f.attrs.modify("model_config", json.dumps(cfg).encode("utf-8"))
            patched += 1
            print("PATCHED:", path)
print(f"Done. Patched {patched} file(s).")
EOF
else
  echo "Skipping H5 patch: directory not found: $H5_ROOT"
fi

echo
echo "==> Setup complete."
echo "Next time, just:"
echo "  conda activate $ENV_NAME"
echo "  python - <<'EOF'"
echo "  import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
echo "  EOF"
