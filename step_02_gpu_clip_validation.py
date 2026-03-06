#!/usr/bin/env python3
"""
V-QMin Replicación — Step 02: Validación GPU + Pipeline CLIP
=============================================================
Confirma GPU + CLIP ViT-B/16 funcionando con imágenes del dataset.
Mide throughput y sanity checks de embeddings.

Entrada:  Parquets en DATASET_DIR/batch_*/
Salida:   OUTPUT_DIR/fase02_gpu_clip.json
"""

import os
import sys
import time
import glob
import json
import numpy as np
from io import BytesIO

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════
DATASET_DIR = r"I:\RIFINALV3"
OUTPUT_DIR = os.path.join(DATASET_DIR, "vqmin_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_TEST_IMAGES = 200
CLIP_BATCH_SIZE = 64

print("=" * 70)
print("V-QMin — STEP 02: VALIDACIÓN GPU + PIPELINE CLIP")
print("=" * 70)

# ══════════════════════════════════════════════════════════════
# PASO 1: VALIDACIÓN DE GPU
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PASO 1: VALIDACIÓN DE GPU")
print("=" * 70)

import torch
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA disponible: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("  ERROR: CUDA no disponible.")
    sys.exit(1)

device = torch.device("cuda:0")
gpu_name = torch.cuda.get_device_name(0)
gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")

t0 = time.time()
x = torch.randn(1000, 1000, device=device)
y = torch.mm(x, x.T)
torch.cuda.synchronize()
print(f"  Test matmul: {time.time()-t0:.3f}s ✓")

# ══════════════════════════════════════════════════════════════
# PASO 2: CARGAR CLIP ViT-B/16
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PASO 2: CARGANDO CLIP ViT-B/16")
print("=" * 70)

import open_clip
print(f"  open_clip: {open_clip.__version__}")

t0 = time.time()
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-16', pretrained='openai', device=device
)
tokenizer = open_clip.get_tokenizer('ViT-B-16')
model.eval()
print(f"  Modelo cargado en {time.time()-t0:.1f}s")
print(f"  Embedding dim: 512")

param_device = next(model.parameters()).device
assert str(param_device).startswith("cuda"), "Modelo no está en GPU"

# ══════════════════════════════════════════════════════════════
# PASO 3: CARGAR IMÁGENES DE PRUEBA
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"PASO 3: CARGANDO {N_TEST_IMAGES} IMÁGENES")
print("=" * 70)

import pyarrow.parquet as pq
from PIL import Image

batch_dirs = sorted(glob.glob(os.path.join(DATASET_DIR, "batch_*")))
all_parquets = []
for bd in batch_dirs:
    all_parquets.extend(sorted(glob.glob(os.path.join(bd, "*.parquet"))))

images_pil = []
t0 = time.time()

for pq_path in all_parquets:
    if len(images_pil) >= N_TEST_IMAGES:
        break
    table = pq.read_table(pq_path)
    df = table.to_pandas()
    for i in range(len(df)):
        if len(images_pil) >= N_TEST_IMAGES:
            break
        try:
            imgs_field = df['images'].iloc[i]
            if hasattr(imgs_field, '__len__') and len(imgs_field) > 0:
                first_img = imgs_field[0]
                img = None
                if isinstance(first_img, Image.Image):
                    img = first_img.convert("RGB")
                elif isinstance(first_img, dict) and 'bytes' in first_img:
                    img = Image.open(BytesIO(first_img['bytes'])).convert("RGB")
                elif isinstance(first_img, bytes):
                    img = Image.open(BytesIO(first_img)).convert("RGB")
                if img is not None:
                    images_pil.append(img)
        except Exception:
            pass

print(f"  Cargadas: {len(images_pil)} en {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════
# PASO 4: CLIP EMBEDDINGS DE IMÁGENES
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"PASO 4: CLIP EMBEDDINGS ({len(images_pil)} imágenes)")
print("=" * 70)

t0 = time.time()
all_img_emb = []

with torch.no_grad():
    for bs in range(0, len(images_pil), CLIP_BATCH_SIZE):
        be = min(bs + CLIP_BATCH_SIZE, len(images_pil))
        batch = images_pil[bs:be]
        tensors = torch.stack([preprocess(img) for img in batch]).to(device)
        feat = model.encode_image(tensors)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        all_img_emb.append(feat.cpu().numpy())

img_embeddings = np.concatenate(all_img_emb, axis=0)
elapsed = time.time() - t0
throughput = len(images_pil) / elapsed

print(f"  Shape: {img_embeddings.shape}")
print(f"  Tiempo: {elapsed:.1f}s")
print(f"  Throughput: {throughput:.0f} img/s")

# ══════════════════════════════════════════════════════════════
# PASO 5: SANITY CHECKS
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PASO 5: SANITY CHECKS")
print("=" * 70)

# Normas
norms = np.linalg.norm(img_embeddings, axis=1)
norm_ok = abs(norms.mean() - 1.0) < 0.01
print(f"  Normas: mean={norms.mean():.4f} {'✓' if norm_ok else '✗'}")

# STEM vs no-STEM
stem_queries = [
    "geometry circle theorem proof with diagram",
    "physics Newton force diagram with vectors",
    "chemical reaction equation balance",
    "calculus derivative integral function graph",
    "statistics probability distribution histogram",
]
non_stem = [
    "cute cat playing with yarn",
    "cooking recipe pasta Italian food",
    "vacation beach tropical island sunset",
]

with torch.no_grad():
    stem_emb = model.encode_text(tokenizer(stem_queries).to(device))
    stem_emb = (stem_emb / stem_emb.norm(dim=-1, keepdim=True)).cpu().numpy()
    nonstem_emb = model.encode_text(tokenizer(non_stem).to(device))
    nonstem_emb = (nonstem_emb / nonstem_emb.norm(dim=-1, keepdim=True)).cpu().numpy()

avg_stem = (img_embeddings @ stem_emb.T).mean()
avg_nonstem = (img_embeddings @ nonstem_emb.T).mean()
sim_ok = avg_stem > avg_nonstem
print(f"  Sim STEM: {avg_stem:.4f}, no-STEM: {avg_nonstem:.4f} {'✓' if sim_ok else '✗'}")

# Diversidad
pw = img_embeddings[:50] @ img_embeddings[:50].T
mask = np.triu(np.ones(pw.shape, dtype=bool), k=1)
pw_vals = pw[mask]
div_ok = pw_vals.std() > 0.05
print(f"  Diversidad (std pairwise): {pw_vals.std():.4f} {'✓' if div_ok else '✗'}")

# NaN/Inf
integrity_ok = np.isnan(img_embeddings).sum() == 0 and np.isinf(img_embeddings).sum() == 0
print(f"  Integridad numérica: {'✓' if integrity_ok else '✗'}")

# ══════════════════════════════════════════════════════════════
# RESUMEN
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("RESUMEN STEP 02")
print("=" * 70)

checks = [
    ("GPU disponible", True),
    ("CLIP en GPU", str(param_device).startswith("cuda")),
    (f"Imágenes: {len(images_pil)}/{N_TEST_IMAGES}", len(images_pil) >= N_TEST_IMAGES),
    ("Normas ~1.0", norm_ok),
    ("STEM > no-STEM", sim_ok),
    ("Diversidad OK", div_ok),
    ("Sin NaN/Inf", integrity_ok),
    (f"Throughput: {throughput:.0f} img/s", throughput > 50),
]

all_pass = True
for name, passed in checks:
    status = "PASS" if passed else "FAIL"
    if not passed: all_pass = False
    print(f"  [{status}] {name}")

results = {
    "gpu": gpu_name, "gpu_mem_gb": round(gpu_mem, 1),
    "pytorch": torch.__version__, "open_clip": open_clip.__version__,
    "throughput": round(throughput, 1), "all_pass": all_pass,
}
rpath = os.path.join(OUTPUT_DIR, "fase02_gpu_clip.json")
with open(rpath, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Guardado: {rpath}")

if all_pass:
    print(f"\nSTEP 02 COMPLETADO ✓ — Siguiente: step_03_extract_targets.py")
else:
    print(f"\nSTEP 02 INCOMPLETO — Revisar FAIL")
