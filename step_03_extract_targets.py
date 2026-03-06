#!/usr/bin/env python3
"""
V-QMin Replicación — Step 03: Extracción de 10K Targets + Embeddings CLIP
==========================================================================
Samplea 10,000 registros con imagen del dataset verified, computa CLIP
embeddings (imagen + texto) con ViT-B/16, y guarda todo para el grafo.

Entrada:  Parquets en DATASET_DIR/batch_*/
Salida:   OUTPUT_DIR/targets_10k.pkl
          OUTPUT_DIR/img_embeddings_10k.npy   (10000, 512)
          OUTPUT_DIR/txt_embeddings_10k.npy   (10000, 512)
"""

import os
import sys
import time
import glob
import json
import pickle
import random
import gc
import numpy as np
import pandas as pd
from io import BytesIO

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════
DATASET_DIR = r"I:\RIFINALV3"
OUTPUT_DIR = os.path.join(DATASET_DIR, "vqmin_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_TARGETS = 10000
CLIP_BATCH_SIZE = 128
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

print("=" * 70)
print(f"V-QMin — STEP 03: EXTRACCIÓN DE {N_TARGETS:,} TARGETS + CLIP EMBEDDINGS")
print("=" * 70)

# ══════════════════════════════════════════════════════════════
# PASO 1: SETUP GPU + CLIP
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PASO 1: SETUP GPU + CLIP ViT-B/16")
print("=" * 70)

import torch
import open_clip
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

t0 = time.time()
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-16', pretrained='openai', device=device
)
tokenizer = open_clip.get_tokenizer('ViT-B-16')
model.eval()
print(f"  CLIP cargado en {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════
# PASO 2: CARGAR METADATOS
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PASO 2: CARGANDO METADATOS")
print("=" * 70)

import pyarrow.parquet as pq

batch_dirs = sorted(glob.glob(os.path.join(DATASET_DIR, "batch_*")))
all_parquets = []
for bd in batch_dirs:
    all_parquets.extend(sorted(glob.glob(os.path.join(bd, "*.parquet"))))
print(f"  Parquets: {len(all_parquets)}")

t0 = time.time()
meta_frames = []
for i, pq_path in enumerate(all_parquets):
    table = pq.read_table(pq_path, columns=['idx', 'question', 'answer', 'difficulty'])
    df = table.to_pandas()
    df['_parquet_idx'] = i
    df['_row_idx'] = range(len(df))
    meta_frames.append(df)
    if (i + 1) % 20 == 0:
        print(f"    {i+1}/{len(all_parquets)} parquets...")

df_meta = pd.concat(meta_frames, ignore_index=True)
print(f"  Total: {len(df_meta):,} registros ({time.time()-t0:.1f}s)")

# ══════════════════════════════════════════════════════════════
# PASO 3: SAMPLEAR 10,000 REGISTROS (estratificado por dificultad)
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"PASO 3: SAMPLEANDO {N_TARGETS:,} REGISTROS")
print("=" * 70)

df_sample = df_meta.groupby('difficulty', group_keys=False).apply(
    lambda x: x.sample(
        n=min(len(x), int(N_TARGETS * len(x) / len(df_meta))),
        random_state=SEED
    )
)

if len(df_sample) < N_TARGETS:
    remaining_idx = df_meta.index.difference(df_sample.index)
    extra = df_meta.loc[remaining_idx].sample(
        n=N_TARGETS - len(df_sample), random_state=SEED
    )
    df_sample = pd.concat([df_sample, extra])

df_sample = df_sample.head(N_TARGETS).reset_index(drop=True)
print(f"  Sampleados: {len(df_sample):,}")
for d in sorted(df_sample['difficulty'].unique()):
    n = (df_sample['difficulty'] == d).sum()
    print(f"    Dificultad {d}: {n:,} ({100*n/len(df_sample):.1f}%)")

# ══════════════════════════════════════════════════════════════
# PASO 4: CARGAR IMÁGENES
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"PASO 4: CARGANDO IMÁGENES DE {len(df_sample):,} REGISTROS")
print("=" * 70)

groups = df_sample.groupby('_parquet_idx')
print(f"  Grupos por parquet: {len(groups)}")

images_pil = []
valid_indices = []
failed = 0

t0 = time.time()
for pq_idx, group in groups:
    pq_path = all_parquets[pq_idx]
    table = pq.read_table(pq_path, columns=['images'])
    df_imgs = table.to_pandas()

    for _, row in group.iterrows():
        try:
            row_idx = row['_row_idx']
            imgs_field = df_imgs['images'].iloc[row_idx]

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
                    valid_indices.append(row.name)
                else:
                    failed += 1
            else:
                failed += 1
        except Exception:
            failed += 1

    loaded = len(images_pil)
    if loaded % 2000 < len(group) or pq_idx == max(groups.groups.keys()):
        print(f"    {loaded:,} cargadas, {failed} fallos ({time.time()-t0:.1f}s)")

print(f"\n  Resultado: {len(images_pil):,} imágenes, {failed} fallos")

df_targets = df_sample.loc[valid_indices].reset_index(drop=True)
print(f"  Targets efectivos: {len(df_targets):,}")

# ══════════════════════════════════════════════════════════════
# PASO 5: CLIP EMBEDDINGS DE IMÁGENES
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"PASO 5: CLIP EMBEDDINGS — {len(images_pil):,} IMÁGENES")
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
        if (bs // CLIP_BATCH_SIZE + 1) % 10 == 0:
            done = be
            print(f"    {done:,}/{len(images_pil):,} ({done/(time.time()-t0):.0f} img/s)")

img_embeddings = np.concatenate(all_img_emb, axis=0)
elapsed_img = time.time() - t0
print(f"\n  Shape: {img_embeddings.shape}")
print(f"  Tiempo: {elapsed_img:.1f}s ({len(images_pil)/elapsed_img:.0f} img/s)")

del images_pil
gc.collect()
torch.cuda.empty_cache()

# ══════════════════════════════════════════════════════════════
# PASO 6: CLIP EMBEDDINGS DE TEXTO
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"PASO 6: CLIP EMBEDDINGS — {len(df_targets):,} TEXTOS")
print("=" * 70)

t0 = time.time()
all_txt_emb = []
questions = df_targets['question'].tolist()

with torch.no_grad():
    for bs in range(0, len(questions), CLIP_BATCH_SIZE):
        be = min(bs + CLIP_BATCH_SIZE, len(questions))
        tokens = tokenizer(questions[bs:be]).to(device)
        feat = model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        all_txt_emb.append(feat.cpu().numpy())

txt_embeddings = np.concatenate(all_txt_emb, axis=0)
elapsed_txt = time.time() - t0
print(f"  Shape: {txt_embeddings.shape}")
print(f"  Tiempo: {elapsed_txt:.1f}s")

# ══════════════════════════════════════════════════════════════
# PASO 7: VALIDACIÓN
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PASO 7: VALIDACIÓN")
print("=" * 70)

sample_n = min(500, len(img_embeddings))
si = np.random.choice(len(img_embeddings), sample_n, replace=False)
same_sim = np.sum(img_embeddings[si] * txt_embeddings[si], axis=1)
shuffle = np.random.permutation(si)
diff_sim = np.sum(img_embeddings[si] * txt_embeddings[shuffle], axis=1)
gap = same_sim.mean() - diff_sim.mean()

print(f"  Coherencia img-txt: same={same_sim.mean():.4f}, diff={diff_sim.mean():.4f}, gap={gap:+.4f}")
print(f"  {'✓ OK' if gap > 0.01 else '⚠ Gap bajo'}")

# ══════════════════════════════════════════════════════════════
# PASO 8: GUARDAR
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PASO 8: GUARDANDO")
print("=" * 70)

df_targets['img_embedding'] = list(img_embeddings)
df_targets['txt_embedding'] = list(txt_embeddings)

pkl_path = os.path.join(OUTPUT_DIR, "targets_10k.pkl")
df_targets.to_pickle(pkl_path)
print(f"  Pickle: {pkl_path} ({os.path.getsize(pkl_path)/1024**2:.1f} MB)")

np.save(os.path.join(OUTPUT_DIR, "img_embeddings_10k.npy"), img_embeddings)
np.save(os.path.join(OUTPUT_DIR, "txt_embeddings_10k.npy"), txt_embeddings)
print(f"  Embeddings: img={img_embeddings.shape}, txt={txt_embeddings.shape}")

results = {
    "n_targets": len(df_targets), "n_failed": failed,
    "img_shape": list(img_embeddings.shape),
    "throughput": round(len(img_embeddings) / elapsed_img, 1),
    "coherence_gap": round(float(gap), 4),
}
with open(os.path.join(OUTPUT_DIR, "fase03_targets.json"), 'w') as f:
    json.dump(results, f, indent=2)

# ══════════════════════════════════════════════════════════════
# RESUMEN
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("RESUMEN STEP 03")
print("=" * 70)

checks = [
    (f"Targets: {len(df_targets):,}/{N_TARGETS:,}", len(df_targets) >= N_TARGETS * 0.99),
    (f"Img embeddings: {img_embeddings.shape}", img_embeddings.shape == (len(df_targets), 512)),
    (f"Txt embeddings: {txt_embeddings.shape}", txt_embeddings.shape == (len(df_targets), 512)),
    (f"Coherencia: {gap:+.4f}", gap > 0.01),
    ("Sin NaN/Inf", np.isnan(img_embeddings).sum() == 0),
]

all_pass = True
for name, passed in checks:
    status = "PASS" if passed else "FAIL"
    if not passed: all_pass = False
    print(f"  [{status}] {name}")

if all_pass:
    print(f"\nSTEP 03 COMPLETADO ✓ — Siguiente: step_04_build_graph.py")
else:
    print(f"\nSTEP 03 INCOMPLETO — Revisar FAIL")
