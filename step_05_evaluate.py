#!/usr/bin/env python3
"""
V-QMin Replicación — Step 05: Crawlers + Evaluación Completa
==============================================================
Ejecuta BFS, Text-Only y V-QMin sobre el grafo de 50K nodos.
Computa HR@K, nDCG@K, alpha sweep y tests de Wilcoxon.

Entrada:  OUTPUT_DIR/synthetic_graph_50k.pkl
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
from collections import deque
from scipy import stats

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════
DATASET_DIR = r"I:\RIFINALV3"
OUTPUT_DIR = os.path.join(DATASET_DIR, "vqmin_outputs")

QUERIES = [
    "geometry circle theorem proof with diagram",
    "physics Newton force diagram with vectors",
    "financial investment compound interest graph",
    "chemical reaction equation balance",
    "electrical circuit engineering design",
    "algebra polynomial equation solve step by step",
    "trigonometry sine cosine angle calculation",
    "calculus derivative integral function graph",
    "statistics probability distribution histogram",
    "optics lens mirror reflection diagram",
]

ALPHA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MAIN_STRATEGIES = [
    ('BFS', None),
    ('Text-Only', 1.0),
    ('V-QMin α=0.2', 0.2),
    ('V-QMin α=0.3', 0.3),
    ('V-QMin α=0.5', 0.5),
]
K_VALUES = [50, 100, 200, 500, 1000, 2000, 5000, 10000]

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("=" * 70)
print("V-QMin — STEP 05: CRAWLERS + EVALUACIÓN COMPLETA")
print("=" * 70)

# ══════════════════════════════════════════════════════════════
# PASO 1: CARGAR GRAFO + CLIP
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PASO 1: CARGANDO GRAFO + CLIP")
print("=" * 70)

import torch
import open_clip

graph_path = os.path.join(OUTPUT_DIR, "synthetic_graph_50k.pkl")
if not os.path.exists(graph_path):
    print(f"  ERROR: {graph_path} no existe. Ejecutar step_04 primero.")
    sys.exit(1)

t0 = time.time()
with open(graph_path, 'rb') as f:
    graph = pickle.load(f)
print(f"  Grafo cargado en {time.time()-t0:.1f}s")

N_TOTAL = graph['n_total']
N_TARGETS = graph['n_targets']
node_types = graph['node_types']
qtext = graph['qtext']
embeddings = graph['embeddings']
seeds = graph['seeds']

# Reconstruir adjacency como dict de sets
adj = {int(k): set(v) for k, v in graph['adjacency'].items()}
target_set = set(range(N_TARGETS))

print(f"  Nodos: {N_TOTAL:,} ({N_TARGETS:,} targets)")
print(f"  Aristas: {sum(len(v) for v in adj.values())//2:,}")

# CLIP para queries
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai', device=device)
tokenizer = open_clip.get_tokenizer('ViT-B-16')
model.eval()

with torch.no_grad():
    tokens = tokenizer(QUERIES).to(device)
    qf = model.encode_text(tokens)
    qf = qf / qf.norm(dim=-1, keepdim=True)
    query_embeddings = qf.cpu().numpy()
print(f"  Query embeddings: {query_embeddings.shape}")

del model; torch.cuda.empty_cache()

# ══════════════════════════════════════════════════════════════
# PASO 2: CRAWLERS
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PASO 2: DEFINIENDO CRAWLERS")
print("=" * 70)

def compute_score(node_id, query_emb, alpha):
    """Score = α·Q_text + (1-α)·Rel_visual, con Rel_visual = (cos+1)/2."""
    q = qtext[node_id]
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

print("  ✓ BFS + Scored crawlers definidos")

# ══════════════════════════════════════════════════════════════
# PASO 3: MÉTRICAS
# ══════════════════════════════════════════════════════════════

def harvest_rate_at_k(visited, ts, k):
    return sum(1 for n in visited[:k] if n in ts) / k if k > 0 else 0.0

def ndcg_at_k(visited, ts, k):
    dcg = sum(1.0 / np.log2(i + 2) for i, n in enumerate(visited[:k]) if n in ts)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(ts))))
    return dcg / idcg if idcg > 0 else 0.0

# ══════════════════════════════════════════════════════════════
# PASO 4: EJECUTAR CRAWLERS
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PASO 3: EJECUTANDO CRAWLERS")
print("=" * 70)

K_MAX = max(K_VALUES)
all_results = {}
t0_total = time.time()

for strategy_name, alpha in MAIN_STRATEGIES:
    all_results[strategy_name] = {}
    for qi, query_text in enumerate(QUERIES):
        t0 = time.time()
        qemb = query_embeddings[qi]
        if alpha is None:
            visited = crawl_bfs(seeds, K_MAX)
        else:
            visited = crawl_scored(seeds, qemb, alpha, K_MAX)

        hr = {k: harvest_rate_at_k(visited, target_set, k) for k in K_VALUES}
        ndcg = {k: ndcg_at_k(visited, target_set, k) for k in K_VALUES}
        all_results[strategy_name][qi] = {'hr': hr, 'ndcg': ndcg}

        print(f"  {strategy_name} | Q{qi+1}/10: HR@500={hr[500]:.3f}, "
              f"nDCG@500={ndcg[500]:.3f} ({time.time()-t0:.1f}s)")
    print()

print(f"  Total: {time.time()-t0_total:.1f}s")

# ══════════════════════════════════════════════════════════════
# PASO 5: ALPHA SWEEP
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PASO 4: ALPHA SWEEP")
print("=" * 70)

alpha_sweep = {}
for alpha in ALPHA_VALUES:
    hr_vals = []
    for qi in range(len(QUERIES)):
        visited = crawl_scored(seeds, query_embeddings[qi], alpha, 500)
        hr_vals.append(harvest_rate_at_k(visited, target_set, 500))
    alpha_sweep[alpha] = {
        'hr_500_mean': float(np.mean(hr_vals)),
        'hr_500_std': float(np.std(hr_vals)),
        'hr_500_per_query': hr_vals,
    }
    print(f"  α={alpha:.1f}: HR@500 = {np.mean(hr_vals):.4f} ± {np.std(hr_vals):.4f}")

best_alpha = max(alpha_sweep.keys(), key=lambda a: alpha_sweep[a]['hr_500_mean'])
print(f"\n  ★ Alfa óptimo: {best_alpha} (HR@500={alpha_sweep[best_alpha]['hr_500_mean']:.4f})")

# ══════════════════════════════════════════════════════════════
# PASO 6: TABLAS + WILCOXON
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("RESULTADOS")
print("=" * 70)

# Tabla HR
print(f"\n{'Estrategia':<25}" + "".join(f"{'HR@'+str(k):>12}" for k in K_VALUES))
print("─" * 120)
for sname, _ in MAIN_STRATEGIES:
    row = f"{sname:<25}"
    for k in K_VALUES:
        vals = [all_results[sname][qi]['hr'][k] for qi in range(len(QUERIES))]
        row += f"{np.mean(vals):>8.3f}±{np.std(vals):.2f}"
    print(row)

# Wilcoxon
main_method = 'V-QMin α=0.2'
print(f"\nWilcoxon signed-rank ({main_method} vs baselines):")
for baseline in ['BFS', 'Text-Only']:
    print(f"\n  vs {baseline}:")
    for k in K_VALUES:
        vq = [all_results[main_method][qi]['hr'][k] for qi in range(len(QUERIES))]
        bs = [all_results[baseline][qi]['hr'][k] for qi in range(len(QUERIES))]
        diffs = [v - b for v, b in zip(vq, bs)]
        if all(d == 0 for d in diffs):
            p = 1.0
        else:
            try:
                _, p = stats.wilcoxon(vq, bs, alternative='greater')
            except Exception:
                p = 1.0
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"    K={k}: p={p:.6f} {sig}")

# Mejora relativa
print(f"\nMejora relativa ({main_method}):")
for baseline in ['BFS', 'Text-Only']:
    for k in [500, 1000, 2000]:
        vq_m = np.mean([all_results[main_method][qi]['hr'][k] for qi in range(len(QUERIES))])
        bs_m = np.mean([all_results[baseline][qi]['hr'][k] for qi in range(len(QUERIES))])
        pct = 100 * (vq_m - bs_m) / bs_m if bs_m > 0 else 0
        print(f"  vs {baseline} HR@{k}: {vq_m:.3f} vs {bs_m:.3f} ({pct:+.1f}%)")

# ══════════════════════════════════════════════════════════════
# PASO 7: GUARDAR
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("GUARDANDO")
print("=" * 70)

export = {}
for sname, _ in MAIN_STRATEGIES:
    export[sname] = {}
    for k in K_VALUES:
        hr_vals = [all_results[sname][qi]['hr'][k] for qi in range(len(QUERIES))]
        ndcg_vals = [all_results[sname][qi]['ndcg'][k] for qi in range(len(QUERIES))]
        export[sname][f'HR@{k}'] = {
            'mean': float(np.mean(hr_vals)), 'std': float(np.std(hr_vals)),
            'values': [float(v) for v in hr_vals],
        }
        export[sname][f'nDCG@{k}'] = {
            'mean': float(np.mean(ndcg_vals)), 'std': float(np.std(ndcg_vals)),
            'values': [float(v) for v in ndcg_vals],
        }

results_full = {
    "n_total": N_TOTAL, "n_targets": N_TARGETS,
    "n_queries": len(QUERIES), "queries": QUERIES,
    "k_values": K_VALUES, "strategies": export,
    "alpha_sweep": {str(a): {'hr_500_mean': v['hr_500_mean'], 'hr_500_std': v['hr_500_std']}
                    for a, v in alpha_sweep.items()},
    "best_alpha": best_alpha,
}

rpath = os.path.join(OUTPUT_DIR, "evaluation_results.json")
with open(rpath, 'w', encoding='utf-8') as f:
    json.dump(results_full, f, indent=2, ensure_ascii=False)
print(f"  {rpath}")

spath = os.path.join(OUTPUT_DIR, "alpha_sweep.json")
with open(spath, 'w', encoding='utf-8') as f:
    json.dump({str(a): v for a, v in alpha_sweep.items()}, f, indent=2)
print(f"  {spath}")

print(f"\nSTEP 05 COMPLETADO ✓ — Siguiente: step_06_figures_main.py")
