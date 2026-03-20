"""
V-QMin — STEP 04: GRAFO SINTÉTICO DE 50K NODOS
================================================
Construye grafo con hard distractors como perturbaciones de targets
individuales (no con centroide).

Entrada:  v2_targets_10k.pkl (targets originales con embeddings ViT-B/16)
Salida:   synthetic_graph_50k.pkl

NOTA: Usa v2_targets_10k.pkl directamente para reproducir los resultados
      exactos del paper.
"""

import os
import sys
import time
import json
import pickle
import random
import numpy as np
import pandas as pd
from collections import Counter, deque

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════
BASE_DIR = r"I:\RIFINALV4"
OUTPUT_DIR = os.path.join(BASE_DIR, "vqmin_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Archivo de targets original (de v2, para reproducibilidad exacta)
TARGETS_PKL = os.path.join(BASE_DIR, "v2_targets_10k.pkl")

N_TARGETS = 10000
DISTRACTOR_RATIO = 4
HARD_RATIO = 0.40
SEED = 42

# Hard distractors: perturbación gaussiana individual
HARD_NOISE_SCALE_T1 = 0.05   # Tier 1: sim ~0.65-0.75
HARD_NOISE_SCALE_T2 = 0.10   # Tier 2: sim ~0.40-0.55
HARD_TIER1_RATIO = 0.50

# Adyacencia
MIN_DEGREE = 3
MAX_DEGREE = 8
P_EDGES = {
    'target': [0.40, 0.30, 0.30],
    'hard':   [0.35, 0.35, 0.30],
    'easy':   [0.15, 0.25, 0.60],
}

# Q_text
QTEXT_HARD_MEAN = 0.70
QTEXT_HARD_STD = 0.08
QTEXT_EASY_MEAN = 0.20
QTEXT_EASY_STD = 0.10

random.seed(SEED)
np.random.seed(SEED)

N_DISTRACTORS = N_TARGETS * DISTRACTOR_RATIO
N_HARD = int(N_DISTRACTORS * HARD_RATIO)
N_EASY = N_DISTRACTORS - N_HARD
N_TOTAL = N_TARGETS + N_DISTRACTORS
N_HARD_T1 = int(N_HARD * HARD_TIER1_RATIO)
N_HARD_T2 = N_HARD - N_HARD_T1

print("=" * 70)
print("V-QMin — STEP 04: GRAFO SINTÉTICO DE 50K NODOS")
print("=" * 70)
print(f"  Targets:       {N_TARGETS:,}")
print(f"  Easy dist.:    {N_EASY:,}")
print(f"  Hard dist.:    {N_HARD:,} (T1={N_HARD_T1:,} σ={HARD_NOISE_SCALE_T1}, T2={N_HARD_T2:,} σ={HARD_NOISE_SCALE_T2})")
print(f"  TOTAL:         {N_TOTAL:,}")

# ══════════════════════════════════════════════════════════════
# PASO 1: CARGAR TARGETS DESDE v2_targets_10k.pkl
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PASO 1: CARGANDO TARGETS (v2_targets_10k.pkl)")
print("=" * 70)

if not os.path.exists(TARGETS_PKL):
    print(f"  ERROR: No se encontró {TARGETS_PKL}")
    print(f"  Copiar v2_targets_10k.pkl desde el directorio v2 original.")
    sys.exit(1)

t0 = time.time()
df_targets = pd.read_pickle(TARGETS_PKL)
img_emb_targets = np.stack(df_targets['img_embedding'].values).astype(np.float32)
print(f"  Targets: {len(df_targets):,} ({time.time()-t0:.1f}s)")
print(f"  Embeddings: {img_emb_targets.shape}")

# Verificar que son los embeddings esperados
assert img_emb_targets.shape == (N_TARGETS, 512), \
    f"Shape inesperado: {img_emb_targets.shape}, esperado ({N_TARGETS}, 512)"

# También guardar embeddings como npy para otros steps
np.save(os.path.join(OUTPUT_DIR, "img_embeddings_10k.npy"), img_emb_targets)

txt_emb_targets = np.stack(df_targets['txt_embedding'].values).astype(np.float32)
np.save(os.path.join(OUTPUT_DIR, "txt_embeddings_10k.npy"), txt_emb_targets)
print(f"  Embeddings npy guardados para otros steps")

# ══════════════════════════════════════════════════════════════
# PASO 2: Q_TEXT PARA TARGETS
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PASO 2: Q_TEXT PARA TARGETS")
print("=" * 70)

def compute_qtext_for_targets(df):
    n = len(df)
    base = np.full(n, 0.50)
    ans_len = df['answer'].str.len().values
    base += np.where(ans_len > 1000, 0.20, ans_len / 1000 * 0.20)
    q_len = df['question'].str.len().values
    base += np.where(q_len > 200, 0.10, q_len / 200 * 0.10)
    diff = df['difficulty'].fillna(2).values.astype(float)
    base += (diff - 1) * 0.02
    noise = np.random.normal(0, 0.03, n)
    return np.clip(base + noise, 0.10, 0.95)

qtext_targets = compute_qtext_for_targets(df_targets)
print(f"  Q_text targets: mean={qtext_targets.mean():.3f}, std={qtext_targets.std():.3f}")

# ══════════════════════════════════════════════════════════════
# PASO 3: HARD DISTRACTORS (MÉTODO CORREGIDO)
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PASO 3: HARD DISTRACTORS (MÉTODO CORREGIDO)")
print("=" * 70)

def make_hard_from_targets(n, target_embs, noise_scale):
    base_indices = np.random.randint(0, len(target_embs), size=n)
    base_embs = target_embs[base_indices]
    noise = np.random.randn(n, 512).astype(np.float32) * noise_scale
    perturbed = base_embs + noise
    norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
    perturbed = perturbed / norms
    return perturbed, base_indices

hard_t1_emb, hard_t1_bases = make_hard_from_targets(N_HARD_T1, img_emb_targets, HARD_NOISE_SCALE_T1)
hard_t2_emb, hard_t2_bases = make_hard_from_targets(N_HARD_T2, img_emb_targets, HARD_NOISE_SCALE_T2)
hard_emb = np.concatenate([hard_t1_emb, hard_t2_emb], axis=0)

# Medir similitud con targets de origen
sim_t1 = np.array([np.dot(hard_t1_emb[i], img_emb_targets[hard_t1_bases[i]])
                    for i in range(min(1000, N_HARD_T1))])
sim_t2 = np.array([np.dot(hard_t2_emb[i], img_emb_targets[hard_t2_bases[i]])
                    for i in range(min(1000, N_HARD_T2))])

print(f"  Tier 1 ({N_HARD_T1:,}, σ={HARD_NOISE_SCALE_T1}): sim={sim_t1.mean():.3f}±{sim_t1.std():.3f}")
print(f"  Tier 2 ({N_HARD_T2:,}, σ={HARD_NOISE_SCALE_T2}): sim={sim_t2.mean():.3f}±{sim_t2.std():.3f}")

qtext_hard = np.clip(np.random.normal(QTEXT_HARD_MEAN, QTEXT_HARD_STD, N_HARD), 0.40, 0.92)
print(f"  Q_text hard: mean={qtext_hard.mean():.3f}")

# ══════════════════════════════════════════════════════════════
# PASO 4: EASY DISTRACTORS
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PASO 4: EASY DISTRACTORS")
print("=" * 70)

target_centroid = img_emb_targets.mean(axis=0)
target_centroid = target_centroid / np.linalg.norm(target_centroid)

easy_emb = np.random.randn(N_EASY, 512).astype(np.float32)
easy_emb = easy_emb * 0.9 + target_centroid * 0.1
easy_emb = easy_emb / np.linalg.norm(easy_emb, axis=1, keepdims=True)

qtext_easy = np.clip(np.random.normal(QTEXT_EASY_MEAN, QTEXT_EASY_STD, N_EASY), 0.05, 0.40)
print(f"  Easy ({N_EASY:,}): Q_text mean={qtext_easy.mean():.3f}")

# ══════════════════════════════════════════════════════════════
# PASO 5: ENSAMBLAR NODOS
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PASO 5: ENSAMBLANDO NODOS")
print("=" * 70)

node_types = ['target'] * N_TARGETS + ['easy'] * N_EASY + ['hard'] * N_HARD
qtext_all = np.concatenate([qtext_targets, qtext_easy, qtext_hard])
all_embeddings = np.concatenate([img_emb_targets, easy_emb, hard_emb], axis=0)

assert len(node_types) == N_TOTAL
assert qtext_all.shape == (N_TOTAL,)
assert all_embeddings.shape == (N_TOTAL, 512)

print(f"  Nodos: {Counter(node_types)}")
print(f"  Embeddings: {all_embeddings.shape} ({all_embeddings.nbytes/1024**2:.1f} MB)")

# Similitudes inter-grupo
target_s = all_embeddings[:500]
easy_s = all_embeddings[N_TARGETS:N_TARGETS+500]
hard_s = all_embeddings[N_TARGETS+N_EASY:N_TARGETS+N_EASY+500]

sim_t_t = target_s @ target_s.T
np.fill_diagonal(sim_t_t, 0)
sim_t_t_mean = sim_t_t.sum() / (500 * 499)
sim_t_h = (target_s @ hard_s.T).mean()
sim_t_e = (target_s @ easy_s.T).mean()

print(f"  Similitudes: T-T={sim_t_t_mean:.4f}, T-H={sim_t_h:.4f}, T-E={sim_t_e:.4f}")
hierarchy_ok = sim_t_t_mean > sim_t_h > sim_t_e
print(f"  {'✓ T-T > T-H > T-E' if hierarchy_ok else '✗ Orden inesperado'}")

# ══════════════════════════════════════════════════════════════
# PASO 6: CONSTRUIR ADYACENCIA
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"PASO 6: CONSTRUYENDO ADYACENCIA ({N_TOTAL:,} nodos)")
print("=" * 70)

t0 = time.time()
idx_target = list(range(0, N_TARGETS))
idx_easy = list(range(N_TARGETS, N_TARGETS + N_EASY))
idx_hard = list(range(N_TARGETS + N_EASY, N_TOTAL))
type_indices = {'target': idx_target, 'easy': idx_easy, 'hard': idx_hard}

adjacency = {i: set() for i in range(N_TOTAL)}

for i in range(N_TOTAL):
    src_type = node_types[i]
    degree = random.randint(MIN_DEGREE, MAX_DEGREE)
    probs = P_EDGES[src_type]

    for _ in range(degree):
        r = random.random()
        if r < probs[0]:
            dest_type = 'target'
        elif r < probs[0] + probs[1]:
            dest_type = 'hard'
        else:
            dest_type = 'easy'

        dest = random.choice(type_indices[dest_type])
        if dest != i:
            adjacency[i].add(dest)
            adjacency[dest].add(i)

    if (i + 1) % 10000 == 0:
        elapsed = time.time() - t0
        print(f"    {i+1:,}/{N_TOTAL:,} ({elapsed:.1f}s)")

n_edges = sum(len(v) for v in adjacency.values()) // 2
degrees = [len(adjacency[i]) for i in range(N_TOTAL)]
avg_degree = np.mean(degrees)
elapsed = time.time() - t0

print(f"\n  Aristas: {n_edges:,}")
print(f"  Grado medio: {avg_degree:.1f}")
print(f"  Tiempo: {elapsed:.1f}s")

# ══════════════════════════════════════════════════════════════
# PASO 7: SEMILLAS
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PASO 7: SEMILLAS")
print("=" * 70)

seeds = []
for stype, pool in [('target', idx_target), ('hard', idx_hard), ('easy', idx_easy)]:
    selected = random.sample(pool, 4)
    seeds.extend(selected)
print(f"  Semillas: {len(seeds)} (4 target + 4 hard + 4 easy)")

# ══════════════════════════════════════════════════════════════
# PASO 8: VALIDACIÓN
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PASO 8: VALIDACIÓN")
print("=" * 70)

visited = set()
queue = deque(seeds)
for s in seeds:
    visited.add(s)
while queue:
    node = queue.popleft()
    for nb in adjacency[node]:
        if nb not in visited:
            visited.add(nb)
            queue.append(nb)

reach_pct = 100 * len(visited) / N_TOTAL
reachable_targets = sum(1 for n in visited if node_types[n] == 'target')
print(f"  Alcanzabilidad: {len(visited):,}/{N_TOTAL:,} ({reach_pct:.1f}%)")
print(f"  Targets alcanzables: {reachable_targets:,}/{N_TARGETS:,}")

# ══════════════════════════════════════════════════════════════
# PASO 9: GUARDAR
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PASO 9: GUARDANDO GRAFO")
print("=" * 70)

graph_data = {
    'n_total': N_TOTAL,
    'n_targets': N_TARGETS,
    'n_easy': N_EASY,
    'n_hard': N_HARD,
    'n_hard_t1': N_HARD_T1,
    'n_hard_t2': N_HARD_T2,
    'node_types': node_types,
    'qtext': qtext_all,
    'embeddings': all_embeddings,
    'adjacency': {k: list(v) for k, v in adjacency.items()},
    'seeds': seeds,
    'target_data': {
        'idx': df_targets['idx'].tolist(),
        'question': df_targets['question'].tolist(),
        'difficulty': df_targets['difficulty'].tolist(),
    },
    'params': {
        'distractor_ratio': DISTRACTOR_RATIO,
        'hard_ratio': HARD_RATIO,
        'hard_noise_scale_t1': HARD_NOISE_SCALE_T1,
        'hard_noise_scale_t2': HARD_NOISE_SCALE_T2,
        'hard_tier1_ratio': HARD_TIER1_RATIO,
        'min_degree': MIN_DEGREE,
        'max_degree': MAX_DEGREE,
        'p_edges': P_EDGES,
        'seed': SEED,
    }
}

graph_path = os.path.join(OUTPUT_DIR, "synthetic_graph_50k.pkl")
t0 = time.time()
with open(graph_path, 'wb') as f:
    pickle.dump(graph_data, f, protocol=pickle.HIGHEST_PROTOCOL)
graph_size = os.path.getsize(graph_path) / 1024**2
print(f"  Grafo: {graph_path} ({graph_size:.1f} MB, {time.time()-t0:.1f}s)")

# ══════════════════════════════════════════════════════════════
# RESUMEN
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("RESUMEN STEP 04")
print("=" * 70)

checks = [
    (f"Total nodos: {N_TOTAL:,}", N_TOTAL == 50000),
    (f"Aristas: {n_edges:,}", n_edges > 200000),
    (f"Grado medio: {avg_degree:.1f}", 4.0 <= avg_degree <= 15.0),
    (f"Alcanzabilidad: {reach_pct:.1f}%", reach_pct > 99),
    (f"T-T > T-H > T-E", hierarchy_ok),
    (f"T-H > 0.15 (hard desafiante)", sim_t_h > 0.15),
]

all_pass = True
for name, passed in checks:
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    print(f"  [{status}] {name}")

if all_pass:
    print(f"\nSTEP 04 COMPLETADO ✓ — Siguiente: step_05_evaluate.py")
else:
    print(f"\nSTEP 04 — HAY PROBLEMAS. Revisar FAIL.")
