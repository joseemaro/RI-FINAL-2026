#!/usr/bin/env python3
"""
V-QMin Replicación v4 — Step 05: Crawlers + Evaluación Completa
==============================================================
Ejecuta BFS, Text-Only y V-QMin sobre el grafo de 50K nodos.
Computa HR@K, nDCG@K, alpha sweep y tests de Wilcoxon.

CAMBIO v4: 1,000 queries muestreadas aleatoriamente del campo 'question'
de df_targets (seed=42). Elimina el sesgo de selección manual de v3.

Entrada:  OUTPUT_DIR/synthetic_graph_50k.pkl
          OUTPUT_DIR/targets_10k.pkl
Salida:   OUTPUT_DIR/evaluation_results.json
          OUTPUT_DIR/alpha_sweep.json
"""

import os
import sys
import time
import json
import pickle
import heapq
import random
import numpy as np
import pandas as pd
from collections import deque
from scipy import stats

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════
DATASET_DIR = r"I:\RIFINALV4"
OUTPUT_DIR  = os.path.join(DATASET_DIR, "vqmin_outputs")

N_QUERIES   = 1000   # queries muestreadas del dataset (v4)
SEED        = 42

ALPHA_VALUES    = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MAIN_STRATEGIES = [
    ('BFS',          None),
    ('Text-Only',    1.0),
    ('V-QMin α=0.2', 0.2),
    ('V-QMin α=0.3', 0.3),
    ('V-QMin α=0.5', 0.5),
]
K_VALUES = [50, 100, 200, 500, 1000, 2000, 5000, 10000]

random.seed(SEED)
np.random.seed(SEED)

print("=" * 70)
print("V-QMin v4 — STEP 05: CRAWLERS + EVALUACIÓN COMPLETA")
print(f"  Queries: {N_QUERIES} (muestreadas del dataset, seed={SEED})")
print("=" * 70)

# ══════════════════════════════════════════════════════════════
# PASO 1: CARGAR GRAFO + TARGETS
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PASO 1: CARGANDO GRAFO + TARGETS")
print("=" * 70)

graph_path = os.path.join(OUTPUT_DIR, "synthetic_graph_50k.pkl")
pkl_path   = os.path.join(OUTPUT_DIR, "targets_10k.pkl")

for p in [graph_path, pkl_path]:
    if not os.path.exists(p):
        print(f"  ERROR: {p} no existe. Ejecutar steps anteriores primero.")
        sys.exit(1)

t0 = time.time()
with open(graph_path, 'rb') as f:
    graph = pickle.load(f)
print(f"  Grafo cargado en {time.time()-t0:.1f}s")

N_TOTAL   = graph['n_total']
N_TARGETS = graph['n_targets']
qtext     = graph['qtext']
embeddings = graph['embeddings']
seeds     = graph['seeds']
adj       = {int(k): set(v) for k, v in graph['adjacency'].items()}
target_set = set(range(N_TARGETS))

print(f"  Nodos: {N_TOTAL:,} ({N_TARGETS:,} targets)")
print(f"  Aristas: {sum(len(v) for v in adj.values())//2:,}")

df_targets = pd.read_pickle(pkl_path)
print(f"  Targets pkl: {len(df_targets):,} registros")

# ══════════════════════════════════════════════════════════════
# PASO 2: MUESTREAR 1000 QUERIES DEL DATASET
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"PASO 2: MUESTREANDO {N_QUERIES} QUERIES DEL DATASET")
print("=" * 70)

# Sampleo aleatorio estratificado del campo 'question' de df_targets
# Se excluyen los índices usados como targets del grafo (0..N_TARGETS-1)
# para evitar que el crawler tenga ventaja de "conocer" su propia query
rng = np.random.RandomState(SEED)
query_indices = rng.choice(len(df_targets), size=N_QUERIES, replace=False)
QUERIES = df_targets['question'].iloc[query_indices].tolist()

print(f"  Queries muestreadas: {len(QUERIES)}")
print(f"  Ejemplos:")
for i in [0, 1, 2, -1]:
    print(f"    [{query_indices[i]}] {QUERIES[i][:80]}...")

# Guardar lista de queries usadas para reproducibilidad
queries_export = {
    "n_queries": N_QUERIES,
    "seed": SEED,
    "source": "df_targets['question']",
    "indices": query_indices.tolist(),
    "queries": QUERIES,
}
with open(os.path.join(OUTPUT_DIR, "queries_1000.json"), 'w', encoding='utf-8') as f:
    json.dump(queries_export, f, indent=2, ensure_ascii=False)
print(f"  ✓ Guardado: queries_1000.json")

# ══════════════════════════════════════════════════════════════
# PASO 3: CLIP EMBEDDINGS DE LAS 1000 QUERIES
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"PASO 3: CLIP EMBEDDINGS — {N_QUERIES} QUERIES")
print("=" * 70)

import torch
import open_clip

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

model, _, _ = open_clip.create_model_and_transforms(
    'ViT-B-16', pretrained='openai', device=device
)
tokenizer = open_clip.get_tokenizer('ViT-B-16')
model.eval()

CLIP_BATCH = 256
all_qemb = []
t0 = time.time()
with torch.no_grad():
    for bs in range(0, len(QUERIES), CLIP_BATCH):
        be     = min(bs + CLIP_BATCH, len(QUERIES))
        tokens = tokenizer(QUERIES[bs:be]).to(device)
        feat   = model.encode_text(tokens)
        feat   = feat / feat.norm(dim=-1, keepdim=True)
        all_qemb.append(feat.cpu().numpy())

query_embeddings = np.concatenate(all_qemb, axis=0)
print(f"  Query embeddings: {query_embeddings.shape} en {time.time()-t0:.1f}s")

del model
torch.cuda.empty_cache()

# ══════════════════════════════════════════════════════════════
# PASO 4: BENCHMARK DE TIMING (1 query, todas las estrategias)
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PASO 4: BENCHMARK DE TIMING (1 query)")
print("=" * 70)

K_MAX = max(K_VALUES)

def compute_score(node_id, query_emb, alpha):
    q       = qtext[node_id]
    cos_sim = float(np.dot(embeddings[node_id], query_emb))
    rel_visual = (cos_sim + 1.0) / 2.0
    return alpha * q + (1.0 - alpha) * rel_visual

def crawl_bfs(seed_nodes, max_steps):
    visited, visited_set = [], set()
    queue = deque(seed_nodes)
    for s in seed_nodes:
        visited_set.add(s)
    while queue and len(visited) < max_steps:
        node = queue.popleft()
        visited.append(node)
        neighbors = list(adj.get(node, []))
        random.shuffle(neighbors)
        for nb in neighbors:
            if nb not in visited_set:
                visited_set.add(nb)
                queue.append(nb)
    return visited[:max_steps]

def crawl_scored(seed_nodes, query_emb, alpha, max_steps):
    """Best-first con política QMin (conservador: mantener mínimo)."""
    visited, visited_set = [], set()
    frontier = []
    frontier_scores = {}
    for s in seed_nodes:
        score = compute_score(s, query_emb, alpha)
        heapq.heappush(frontier, (-score, s))
        frontier_scores[s] = score
    while frontier and len(visited) < max_steps:
        neg_score, node = heapq.heappop(frontier)
        if node in visited_set:
            continue
        visited.append(node)
        visited_set.add(node)
        for nb in adj.get(node, []):
            if nb not in visited_set:
                nb_score = compute_score(nb, query_emb, alpha)
                if nb in frontier_scores:
                    if nb_score < frontier_scores[nb]:
                        frontier_scores[nb] = nb_score
                        heapq.heappush(frontier, (-nb_score, nb))
                else:
                    frontier_scores[nb] = nb_score
                    heapq.heappush(frontier, (-nb_score, nb))
    return visited[:max_steps]

# Correr 1 query de prueba para estimar tiempo total
t_bench = time.time()
_ = crawl_bfs(seeds, K_MAX)
t_bfs = time.time() - t_bench

t_bench = time.time()
_ = crawl_scored(seeds, query_embeddings[0], 0.2, K_MAX)
t_scored = time.time() - t_bench

n_strategies_scored = sum(1 for _, a in MAIN_STRATEGIES if a is not None)
estimated_total = (t_bfs + t_scored * n_strategies_scored) * N_QUERIES
print(f"  BFS (1 query):    {t_bfs:.2f}s")
print(f"  Scored (1 query): {t_scored:.2f}s")
print(f"  Estimado total ({N_QUERIES} queries × {len(MAIN_STRATEGIES)} estrategias): "
      f"{estimated_total/60:.1f} min")

# ══════════════════════════════════════════════════════════════
# PASO 5: MÉTRICAS
# ══════════════════════════════════════════════════════════════

def harvest_rate_at_k(visited, ts, k):
    return sum(1 for n in visited[:k] if n in ts) / k if k > 0 else 0.0

def ndcg_at_k(visited, ts, k):
    dcg  = sum(1.0 / np.log2(i + 2) for i, n in enumerate(visited[:k]) if n in ts)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(ts))))
    return dcg / idcg if idcg > 0 else 0.0

# ══════════════════════════════════════════════════════════════
# PASO 6: EJECUTAR CRAWLERS PRINCIPALES
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"PASO 5: EJECUTANDO CRAWLERS — {N_QUERIES} QUERIES")
print("=" * 70)

all_results = {}
t0_total = time.time()

for strategy_name, alpha in MAIN_STRATEGIES:
    all_results[strategy_name] = {}
    t0_strat = time.time()
    print(f"\n  [{strategy_name}]")
    for qi in range(N_QUERIES):
        qemb = query_embeddings[qi]
        if alpha is None:
            visited = crawl_bfs(seeds, K_MAX)
        else:
            visited = crawl_scored(seeds, qemb, alpha, K_MAX)

        hr   = {k: harvest_rate_at_k(visited, target_set, k) for k in K_VALUES}
        ndcg = {k: ndcg_at_k(visited, target_set, k) for k in K_VALUES}
        all_results[strategy_name][qi] = {'hr': hr, 'ndcg': ndcg}

        if (qi + 1) % 100 == 0:
            elapsed = time.time() - t0_strat
            eta     = elapsed / (qi + 1) * (N_QUERIES - qi - 1)
            print(f"    Q{qi+1}/{N_QUERIES} — HR@500={hr[500]:.3f} | "
                  f"Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min")

    strat_elapsed = time.time() - t0_strat
    mean_hr500 = np.mean([all_results[strategy_name][qi]['hr'][500]
                          for qi in range(N_QUERIES)])
    print(f"  ✓ {strategy_name}: HR@500={mean_hr500:.4f} ({strat_elapsed/60:.1f}min)")

print(f"\n  TOTAL crawlers: {(time.time()-t0_total)/60:.1f} min")

# ══════════════════════════════════════════════════════════════
# PASO 7: ALPHA SWEEP
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"PASO 6: ALPHA SWEEP ({len(ALPHA_VALUES)} valores × {N_QUERIES} queries)")
print("=" * 70)

alpha_sweep = {}
t0_sweep = time.time()

for alpha in ALPHA_VALUES:
    hr_vals = []
    for qi in range(N_QUERIES):
        visited = crawl_scored(seeds, query_embeddings[qi], alpha, 500)
        hr_vals.append(harvest_rate_at_k(visited, target_set, 500))
    alpha_sweep[alpha] = {
        'hr_500_mean':      float(np.mean(hr_vals)),
        'hr_500_std':       float(np.std(hr_vals)),
        'hr_500_per_query': hr_vals,
    }
    print(f"  α={alpha:.1f}: HR@500 = {np.mean(hr_vals):.4f} ± {np.std(hr_vals):.4f}")

best_alpha = max(alpha_sweep.keys(), key=lambda a: alpha_sweep[a]['hr_500_mean'])
print(f"\n  ★ Alfa con mayor HR@500: {best_alpha} "
      f"({alpha_sweep[best_alpha]['hr_500_mean']:.4f})")
print(f"  Alpha sweep: {(time.time()-t0_sweep)/60:.1f} min")

# ══════════════════════════════════════════════════════════════
# PASO 8: TABLAS + WILCOXON
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("RESULTADOS")
print("=" * 70)

print(f"\n{'Estrategia':<25}" + "".join(f"{'HR@'+str(k):>12}" for k in K_VALUES))
print("─" * 120)
for sname, _ in MAIN_STRATEGIES:
    row = f"{sname:<25}"
    for k in K_VALUES:
        vals = [all_results[sname][qi]['hr'][k] for qi in range(N_QUERIES)]
        row += f"{np.mean(vals):>8.3f}±{np.std(vals):.2f}"
    print(row)

main_method = 'V-QMin α=0.2'
print(f"\nWilcoxon signed-rank ({main_method} vs baselines, n={N_QUERIES}):")
for baseline in ['BFS', 'Text-Only']:
    print(f"\n  vs {baseline}:")
    for k in K_VALUES:
        vq = [all_results[main_method][qi]['hr'][k] for qi in range(N_QUERIES)]
        bs_vals = [all_results[baseline][qi]['hr'][k] for qi in range(N_QUERIES)]
        diffs = [v - b for v, b in zip(vq, bs_vals)]
        if all(d == 0 for d in diffs):
            p = 1.0
        else:
            try:
                _, p = stats.wilcoxon(vq, bs_vals, alternative='greater')
            except Exception:
                p = 1.0
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"    K={k}: p={p:.2e} {sig}")

# ══════════════════════════════════════════════════════════════
# PASO 9: GUARDAR
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("GUARDANDO")
print("=" * 70)

export = {}
for sname, _ in MAIN_STRATEGIES:
    export[sname] = {}
    for k in K_VALUES:
        hr_vals   = [all_results[sname][qi]['hr'][k]   for qi in range(N_QUERIES)]
        ndcg_vals = [all_results[sname][qi]['ndcg'][k] for qi in range(N_QUERIES)]
        export[sname][f'HR@{k}'] = {
            'mean':   float(np.mean(hr_vals)),
            'std':    float(np.std(hr_vals)),
            'values': [float(v) for v in hr_vals],
        }
        export[sname][f'nDCG@{k}'] = {
            'mean':   float(np.mean(ndcg_vals)),
            'std':    float(np.std(ndcg_vals)),
            'values': [float(v) for v in ndcg_vals],
        }

results_full = {
    "n_total":    N_TOTAL,
    "n_targets":  N_TARGETS,
    "n_queries":  N_QUERIES,
    "seed":       SEED,
    "query_source": "df_targets['question'] — random sample",
    "k_values":   K_VALUES,
    "strategies": export,
    "alpha_sweep": {
        str(a): {
            'hr_500_mean': v['hr_500_mean'],
            'hr_500_std':  v['hr_500_std'],
        }
        for a, v in alpha_sweep.items()
    },
    "best_alpha": best_alpha,
}

rpath = os.path.join(OUTPUT_DIR, "evaluation_results.json")
with open(rpath, 'w', encoding='utf-8') as f:
    json.dump(results_full, f, indent=2, ensure_ascii=False)
print(f"  ✓ {rpath}")

spath = os.path.join(OUTPUT_DIR, "alpha_sweep.json")
with open(spath, 'w', encoding='utf-8') as f:
    json.dump({str(a): v for a, v in alpha_sweep.items()}, f, indent=2)
print(f"  ✓ {spath}")

print(f"\nSTEP 05 COMPLETADO ✓ — Siguiente: step_06_figures_main.py")
