#!/usr/bin/env python3
"""
V-QMin Replicación — Step 01: Exploración del Dataset
======================================================
Valida que los parquets locales de VisualWebInstruct-verified se cargan
correctamente, que las imágenes están incrustadas y son accesibles,
y reporta estadísticas del dataset.

Entrada:  Parquets en DATASET_DIR/batch_XXXX_of_0022/
Salida:   OUTPUT_DIR/fase01_exploration.json
"""

import os
import sys
import time
import glob
import json
from collections import Counter

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════
DATASET_DIR = r"I:\RIFINALV3"
OUTPUT_DIR = os.path.join(DATASET_DIR, "vqmin_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("V-QMin — STEP 01: EXPLORACIÓN DEL DATASET VERIFIED")
print("=" * 70)
print(f"  Dataset dir: {DATASET_DIR}")
print(f"  Output dir:  {OUTPUT_DIR}")

# ══════════════════════════════════════════════════════════════
# PASO 1: VERIFICAR ESTRUCTURA DE ARCHIVOS
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PASO 1: VERIFICANDO ESTRUCTURA DE ARCHIVOS")
print("=" * 70)

batch_dirs = sorted(glob.glob(os.path.join(DATASET_DIR, "batch_*")))
print(f"  Carpetas batch encontradas: {len(batch_dirs)}")

if len(batch_dirs) == 0:
    print("  ERROR: No se encontraron carpetas batch_*")
    print(f"  Contenido de {DATASET_DIR}:")
    for item in os.listdir(DATASET_DIR)[:20]:
        print(f"    {item}")
    sys.exit(1)

total_parquets = 0
for bd in batch_dirs:
    batch_name = os.path.basename(bd)
    parquets = sorted(glob.glob(os.path.join(bd, "*.parquet")))
    total_size_mb = sum(os.path.getsize(p) for p in parquets) / 1024**2
    total_parquets += len(parquets)
    print(f"  {batch_name}: {len(parquets)} parquet(s), {total_size_mb:.1f} MB")

print(f"\n  Total: {len(batch_dirs)} batches, {total_parquets} archivos parquet")

expected_batches = [f"batch_{i:04d}_of_0022" for i in range(1, 23)]
missing = [b for b in expected_batches if not os.path.isdir(os.path.join(DATASET_DIR, b))]
if missing:
    print(f"  Batches ausentes: {missing}")
    print(f"  (Nota: batch_0006 y batch_0007 faltan según documentación oficial)")

# ══════════════════════════════════════════════════════════════
# PASO 2: CARGAR UN BATCH DE PRUEBA
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PASO 2: CARGANDO BATCH DE PRUEBA")
print("=" * 70)

import pandas as pd
import pyarrow.parquet as pq

first_batch_dir = batch_dirs[0]
first_parquet = sorted(glob.glob(os.path.join(first_batch_dir, "*.parquet")))[0]
print(f"  Leyendo: {first_parquet}")

t0 = time.time()
pf = pq.read_table(first_parquet)
print(f"  Schema:")
for field in pf.schema:
    print(f"    {field.name}: {field.type}")
print(f"  Filas: {len(pf)}")
print(f"  Lectura: {time.time()-t0:.1f}s")

df_sample = pf.to_pandas()
print(f"  Columnas: {list(df_sample.columns)}")

# ══════════════════════════════════════════════════════════════
# PASO 3: VERIFICAR CAMPO IMAGES
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PASO 3: VERIFICANDO CAMPO 'images'")
print("=" * 70)

if 'images' not in df_sample.columns:
    print("  ERROR: No se encontró columna 'images'")
    sys.exit(1)

from PIL import Image
from io import BytesIO

sample_images_field = df_sample['images'].iloc[0]
print(f"  Tipo del campo 'images': {type(sample_images_field)}")

if isinstance(sample_images_field, (list, tuple)) and len(sample_images_field) > 0:
    first_img = sample_images_field[0]
    print(f"  Tipo del primer elemento: {type(first_img)}")
    if isinstance(first_img, Image.Image):
        print(f"  ✓ PIL.Image directamente — {first_img.size}, {first_img.mode}")
    elif isinstance(first_img, dict) and 'bytes' in first_img:
        img = Image.open(BytesIO(first_img['bytes'])).convert("RGB")
        print(f"  ✓ Dict con bytes — {img.size}")
    elif isinstance(first_img, bytes):
        img = Image.open(BytesIO(first_img)).convert("RGB")
        print(f"  ✓ Bytes directos — {img.size}")

# ══════════════════════════════════════════════════════════════
# PASO 4: TEST DE CARGA DE IMÁGENES (100 filas)
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PASO 4: TEST DE CARGA DE IMÁGENES (100 filas)")
print("=" * 70)

n_test = min(100, len(df_sample))
loaded = 0
failed = 0
img_sizes = []

for i in range(n_test):
    try:
        imgs_field = df_sample['images'].iloc[i]
        if hasattr(imgs_field, '__len__') and len(imgs_field) > 0:
            first = imgs_field[0]
            img = None
            if isinstance(first, Image.Image):
                img = first.convert("RGB")
            elif isinstance(first, dict) and 'bytes' in first:
                img = Image.open(BytesIO(first['bytes'])).convert("RGB")
            elif isinstance(first, bytes):
                img = Image.open(BytesIO(first)).convert("RGB")
            if img is not None:
                loaded += 1
                img_sizes.append(img.size)
            else:
                failed += 1
        else:
            failed += 1
    except Exception:
        failed += 1

print(f"  Probadas: {n_test}")
print(f"  ✓ Cargadas: {loaded} ({100*loaded/n_test:.1f}%)")
print(f"  ✗ Fallidas: {failed}")

# ══════════════════════════════════════════════════════════════
# PASO 5: ESTADÍSTICAS DEL DATASET COMPLETO
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PASO 5: ESTADÍSTICAS DEL DATASET COMPLETO")
print("=" * 70)

all_parquets = []
for bd in batch_dirs:
    all_parquets.extend(sorted(glob.glob(os.path.join(bd, "*.parquet"))))

t0 = time.time()
total_rows = 0
difficulty_counts = Counter()
question_lengths = []
answer_lengths = []

for i, pq_path in enumerate(all_parquets):
    try:
        table = pq.read_table(pq_path, columns=['idx', 'question', 'answer', 'difficulty'])
        df_batch = table.to_pandas()
        total_rows += len(df_batch)
        if 'difficulty' in df_batch.columns:
            difficulty_counts.update(df_batch['difficulty'].value_counts().to_dict())
        question_lengths.extend(df_batch['question'].str.len().tolist())
        answer_lengths.extend(df_batch['answer'].str.len().tolist())
        if (i + 1) % 10 == 0 or i == len(all_parquets) - 1:
            print(f"    {i+1}/{len(all_parquets)} parquets ({total_rows:,} filas)")
    except Exception as e:
        print(f"    Error en {pq_path}: {e}")

print(f"\n  Total registros: {total_rows:,}")
print(f"  Tiempo: {time.time()-t0:.1f}s")

if difficulty_counts:
    print(f"\n  Distribución de dificultad:")
    for d in sorted(difficulty_counts.keys()):
        count = difficulty_counts[d]
        print(f"    Dificultad {d}: {count:,} ({100*count/total_rows:.1f}%)")

if question_lengths:
    q_lens = sorted(question_lengths)
    a_lens = sorted(answer_lengths)
    print(f"\n  Largo preguntas: min={q_lens[0]}, median={q_lens[len(q_lens)//2]}, "
          f"max={q_lens[-1]}, mean={sum(q_lens)/len(q_lens):.0f}")
    print(f"  Largo respuestas: min={a_lens[0]}, median={a_lens[len(a_lens)//2]}, "
          f"max={a_lens[-1]}, mean={sum(a_lens)/len(a_lens):.0f}")

# ══════════════════════════════════════════════════════════════
# RESUMEN
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("RESUMEN STEP 01")
print("=" * 70)

results = {
    "dataset": "VisualWebInstruct-verified",
    "dataset_dir": DATASET_DIR,
    "total_batches": len(batch_dirs),
    "total_parquets": total_parquets,
    "total_rows": total_rows,
    "missing_batches": missing,
    "image_load_test": {"tested": n_test, "loaded": loaded, "failed": failed},
    "difficulty_distribution": {str(k): v for k, v in sorted(difficulty_counts.items())},
}

checks = [
    ("Batches encontrados (≥18)", len(batch_dirs) >= 18),
    ("Parquets encontrados", total_parquets > 0),
    (f"Total registros ({total_rows:,} ≥ 90K)", total_rows > 90000),
    (f"Imágenes cargables ({loaded}/{n_test} ≥ 95%)", loaded >= n_test * 0.95),
]

all_pass = True
for name, passed in checks:
    status = "PASS" if passed else "FAIL"
    if not passed: all_pass = False
    print(f"  [{status}] {name}")

results_path = os.path.join(OUTPUT_DIR, "fase01_exploration.json")
with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n  Guardado: {results_path}")

if all_pass:
    print(f"\nSTEP 01 COMPLETADO ✓ — Siguiente: step_02_gpu_clip_validation.py")
else:
    print(f"\nSTEP 01 INCOMPLETO — Revisar FAIL arriba")
