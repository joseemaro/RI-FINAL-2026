"""
V-QMin — STEP 07: EMBEDDINGS ViT-L/14 PARA ABLACIÓN
=====================================================
Genera embeddings CLIP con ViT-L/14 para las mismas 10,000 imágenes,
permitiendo comparar ViT-B/16 vs ViT-L/14 en Experimento 1.

Entrada:  v2_targets_10k.pkl (contiene _parquet_idx y _row_idx)
          Dataset parquets en I:\RIFINALV4\ (para re-cargar imágenes)
Salida:   vitl14_img_embeddings_10k.npy
          vitl14_txt_embeddings_10k.npy

NOTA: Re-carga imágenes del dataset porque el pkl solo guarda
      embeddings ViT-B/16, no las imágenes PIL.
"""

import os
import sys
import time
import glob
import pickle
import random
import numpy as np
import pandas as pd
from pathlib import Path
from io import BytesIO

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════
BASE_DIR = r"I:\RIFINALV4"
DATASET_DIR = BASE_DIR                    
OUTPUT_DIR = os.path.join(BASE_DIR, "vqmin_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGETS_PKL = os.path.join(BASE_DIR, "v2_targets_10k.pkl")

CLIP_MODEL = 'ViT-L-14'
CLIP_PRETRAINED = 'openai'
CLIP_BATCH_SIZE = 64        
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

print("=" * 70)
print(f"V-QMin — STEP 07: EMBEDDINGS {CLIP_MODEL}")
print("=" * 70)

# ══════════════════════════════════════════════════════════════
# PASO 1: CARGAR METADATOS DE TARGETS
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PASO 1: CARGANDO TARGETS (metadatos)")
print("=" * 70)

if not os.path.exists(TARGETS_PKL):
    print(f"  ERROR: No se encontró {TARGETS_PKL}")
    sys.exit(1)

df_targets = pd.read_pickle(TARGETS_PKL)
print(f"  Targets: {len(df_targets):,}")

# Verificar que tiene las columnas necesarias para re-cargar imágenes
assert '_parquet_idx' in df_targets.columns, "Falta _parquet_idx en pkl"
assert '_row_idx' in df_targets.columns, "Falta _row_idx en pkl"
print(f"  Columnas: {list(df_targets.columns)}")

# ══════════════════════════════════════════════════════════════
# PASO 2: SETUP GPU + CLIP ViT-L/14
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"PASO 2: SETUP GPU + CLIP {CLIP_MODEL}")
print("=" * 70)

import torch
import open_clip
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

t0 = time.time()
model, _, preprocess = open_clip.create_model_and_transforms(
    CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=device
)
tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
model.eval()

# Verificar dimensionalidad
with torch.no_grad():
    dummy = tokenizer(["test"]).to(device)
    dummy_feat = model.encode_text(dummy)
    EMBED_DIM = dummy_feat.shape[1]

print(f"  Modelo cargado en {time.time()-t0:.1f}s")
print(f"  Dimensión embeddings: {EMBED_DIM}")
print(f"  Batch size: {CLIP_BATCH_SIZE}")

# ══════════════════════════════════════════════════════════════
# PASO 3: RE-CARGAR IMÁGENES DE LOS MISMOS 10,000 REGISTROS
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"PASO 3: RE-CARGANDO IMÁGENES DE {len(df_targets):,} REGISTROS")
print("=" * 70)

import pyarrow.parquet as pq

# Los parquets están en DATASET_DIR/batch_XXXX_of_0022/
batch_dirs = sorted(glob.glob(os.path.join(DATASET_DIR, "batch_*")))
all_parquets = []
for bd in batch_dirs:
    all_parquets.extend(sorted(glob.glob(os.path.join(bd, "*.parquet"))))
print(f"  Parquets disponibles: {len(all_parquets)}")

# Agrupar por parquet para carga eficiente
groups = df_targets.groupby('_parquet_idx')
print(f"  Grupos por parquet: {len(groups)}")

images_pil = []
valid_mask = []
failed = 0

t0 = time.time()
for pq_idx, group in groups:
    pq_path = all_parquets[pq_idx]
    
    # Cargar solo columna images de este parquet
    table = pq.read_table(pq_path, columns=['images'])
    df_imgs = table.to_pandas()
    
    for _, row in group.iterrows():
        try:
            row_idx = row['_row_idx']
            imgs_field = df_imgs['images'].iloc[row_idx]
            img = None
            
            if hasattr(imgs_field, '__len__') and len(imgs_field) > 0:
                first_img = imgs_field[0]
                if isinstance(first_img, Image.Image):
                    img = first_img.convert("RGB")
                elif isinstance(first_img, dict) and 'bytes' in first_img:
                    img = Image.open(BytesIO(first_img['bytes'])).convert("RGB")
                elif isinstance(first_img, bytes):
                    img = Image.open(BytesIO(first_img)).convert("RGB")
            
            if img is not None:
                images_pil.append(img)
                valid_mask.append(True)
            else:
                valid_mask.append(False)
                failed += 1
        except Exception as e:
            valid_mask.append(False)
            failed += 1
    
    loaded_so_far = len(images_pil)
    if loaded_so_far % 2000 < len(group) or pq_idx == max(groups.groups.keys()):
        elapsed = time.time() - t0
        print(f"    {loaded_so_far:,} cargadas, {failed} fallos ({elapsed:.1f}s)")

elapsed = time.time() - t0
print(f"\n  Resultado: {len(images_pil):,} imágenes, {failed} fallos ({elapsed:.1f}s)")

if failed > len(df_targets) * 0.01:
    print(f"  ADVERTENCIA: {failed} fallos ({100*failed/len(df_targets):.1f}%). Verificar paths.")

# ══════════════════════════════════════════════════════════════
# PASO 4: CLIP ViT-L/14 EMBEDDINGS — IMÁGENES
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"PASO 4: CLIP {CLIP_MODEL} EMBEDDINGS — {len(images_pil):,} IMÁGENES")
print("=" * 70)

t0 = time.time()
all_img_emb = []

with torch.no_grad():
    for batch_start in range(0, len(images_pil), CLIP_BATCH_SIZE):
        batch_end = min(batch_start + CLIP_BATCH_SIZE, len(images_pil))
        batch_imgs = images_pil[batch_start:batch_end]

        batch_tensors = torch.stack([preprocess(img) for img in batch_imgs]).to(device)
        img_features = model.encode_image(batch_tensors)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        all_img_emb.append(img_features.cpu().numpy())

        if (batch_start // CLIP_BATCH_SIZE + 1) % 10 == 0:
            done = batch_end
            print(f"    {done:,}/{len(images_pil):,} "
                  f"({100*done/len(images_pil):.0f}%, "
                  f"{done/(time.time()-t0):.0f} img/s)")

img_embeddings = np.concatenate(all_img_emb, axis=0)
elapsed_img = time.time() - t0
print(f"\n  Image embeddings: {img_embeddings.shape}")
print(f"  Tiempo: {elapsed_img:.1f}s ({len(images_pil)/elapsed_img:.0f} img/s)")

# Liberar memoria
del images_pil
import gc; gc.collect()
torch.cuda.empty_cache()

# ══════════════════════════════════════════════════════════════
# PASO 5: CLIP ViT-L/14 EMBEDDINGS — TEXTO
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"PASO 5: CLIP {CLIP_MODEL} EMBEDDINGS — TEXTO")
print("=" * 70)

t0 = time.time()
all_txt_emb = []
questions = df_targets['question'].tolist()

with torch.no_grad():
    for batch_start in range(0, len(questions), CLIP_BATCH_SIZE):
        batch_end = min(batch_start + CLIP_BATCH_SIZE, len(questions))
        batch_texts = questions[batch_start:batch_end]

        tokens = tokenizer(batch_texts).to(device)
        txt_features = model.encode_text(tokens)
        txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)
        all_txt_emb.append(txt_features.cpu().numpy())

txt_embeddings = np.concatenate(all_txt_emb, axis=0)
elapsed_txt = time.time() - t0
print(f"  Text embeddings: {txt_embeddings.shape}")
print(f"  Tiempo: {elapsed_txt:.1f}s")

# ══════════════════════════════════════════════════════════════
# PASO 6: VALIDACIÓN Y COMPARACIÓN CON ViT-B/16
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PASO 6: VALIDACIÓN Y COMPARACIÓN")
print("=" * 70)

# Cargar embeddings ViT-B/16 para comparación
vitb_img = np.load(os.path.join(OUTPUT_DIR, "img_embeddings_10k.npy"))
vitb_txt = np.load(os.path.join(OUTPUT_DIR, "txt_embeddings_10k.npy"))

print(f"  ViT-B/16: img={vitb_img.shape}, txt={vitb_txt.shape}")
print(f"  ViT-L/14: img={img_embeddings.shape}, txt={txt_embeddings.shape}")

# Coherencia imagen-texto
sample_n = min(500, len(img_embeddings))
sample_idx = np.random.choice(len(img_embeddings), sample_n, replace=False)

# ViT-L/14
same_sim_L = np.sum(img_embeddings[sample_idx] * txt_embeddings[sample_idx], axis=1)
shuffle = np.random.permutation(sample_idx)
diff_sim_L = np.sum(img_embeddings[sample_idx] * txt_embeddings[shuffle], axis=1)
gap_L = same_sim_L.mean() - diff_sim_L.mean()

# ViT-B/16
same_sim_B = np.sum(vitb_img[sample_idx] * vitb_txt[sample_idx], axis=1)
diff_sim_B = np.sum(vitb_img[sample_idx] * vitb_txt[shuffle], axis=1)
gap_B = same_sim_B.mean() - diff_sim_B.mean()

print(f"\n  Coherencia imagen-texto:")
print(f"    ViT-B/16: same={same_sim_B.mean():.4f}, diff={diff_sim_B.mean():.4f}, gap={gap_B:.4f}")
print(f"    ViT-L/14: same={same_sim_L.mean():.4f}, diff={diff_sim_L.mean():.4f}, gap={gap_L:.4f}")

# ══════════════════════════════════════════════════════════════
# PASO 7: GUARDAR
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PASO 7: GUARDANDO")
print("=" * 70)

np.save(os.path.join(OUTPUT_DIR, "vitl14_img_embeddings_10k.npy"), img_embeddings)
np.save(os.path.join(OUTPUT_DIR, "vitl14_txt_embeddings_10k.npy"), txt_embeddings)

import json
results = {
    "model": CLIP_MODEL,
    "pretrained": CLIP_PRETRAINED,
    "embed_dim": int(EMBED_DIM),
    "n_images": len(img_embeddings),
    "throughput_img_per_sec": round(len(img_embeddings) / elapsed_img, 1),
    "time_img_sec": round(elapsed_img, 1),
    "time_txt_sec": round(elapsed_txt, 1),
    "coherence_gap_vitl14": round(float(gap_L), 4),
    "coherence_gap_vitb16": round(float(gap_B), 4),
}
with open(os.path.join(OUTPUT_DIR, "step07_vitl14.json"), 'w') as f:
    json.dump(results, f, indent=2)

print(f"  vitl14_img_embeddings_10k.npy ({img_embeddings.shape})")
print(f"  vitl14_txt_embeddings_10k.npy ({txt_embeddings.shape})")
print(f"  Throughput ViT-L/14: {len(img_embeddings)/elapsed_img:.0f} img/s")

# ══════════════════════════════════════════════════════════════
# RESUMEN
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("RESUMEN STEP 07")
print("=" * 70)

checks = [
    (f"Imágenes cargadas: {sum(valid_mask):,}", sum(valid_mask) >= len(df_targets) * 0.99),
    (f"Img embeddings: {img_embeddings.shape}", img_embeddings.shape[1] == EMBED_DIM),
    (f"Txt embeddings: {txt_embeddings.shape}", txt_embeddings.shape[1] == EMBED_DIM),
    (f"Coherencia ViT-L/14: gap={gap_L:.4f}", gap_L > 0.01),
    ("Sin NaN/Inf", np.isnan(img_embeddings).sum() == 0 and np.isnan(txt_embeddings).sum() == 0),
]

all_pass = True
for name, passed in checks:
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    print(f"  [{status}] {name}")

if all_pass:
    print(f"\nSTEP 07 COMPLETADO ✓ — Siguiente: step_08_complementary_experiments.py")
else:
    print(f"\nSTEP 07 — HAY PROBLEMAS. Revisar FAIL.")
