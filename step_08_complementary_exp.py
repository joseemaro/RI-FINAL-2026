#!/usr/bin/env python3
"""
V-QMin — STEP 08: TRES EXPERIMENTOS NUEVOS
=============================================
Experimento 1: Comparacion CLIP — ViT-B/16 vs ViT-L/14 (mismo grafo)
Experimento 2: Robustez — Variar ratio hard distractors (20%, 40%, 60%, 80%)
Experimento 3: Escalado queries — n=10, 30, 50

Entradas:
  - targets_10k.pkl (targets con metadatos)
  - img_embeddings_10k.npy (ViT-B/16)
  - v3_vitl14_img_embeddings_10k.npy (ViT-L/14, de Step 1)

Salidas:
  - v3_exp1_clip_ablation.json
  - v3_exp2_distractor_robustness.json
  - v3_exp3_query_scaling.json
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
from collections import Counter, deque, defaultdict
from scipy import stats

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════
OUTPUT_DIR = os.path.join(r"I:\RIFINALV3", "vqmin_outputs")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_TARGETS = 10000
K_VALUES = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
K_MAX = max(K_VALUES)

# 50 queries STEM para máxima cobertura
QUERIES_50 = [
    # 10 originales de v2
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
    # 20 adicionales para n=30
    "thermodynamics heat transfer conduction convection",
    "biology cell division mitosis meiosis diagram",
    "astronomy planetary orbit Kepler laws",
    "linear algebra matrix determinant eigenvalue",
    "differential equations solution method",
    "quantum mechanics wave function Schrodinger",
    "organic chemistry molecular structure benzene",
    "fluid mechanics Bernoulli equation flow",
    "number theory prime factorization proof",
    "computer science algorithm complexity graph",
    "electromagnetic wave Maxwell equations",
    "structural engineering stress strain diagram",
    "discrete mathematics combinatorics permutation",
    "environmental science ecosystem food chain",
    "materials science crystal structure lattice",
    "control systems transfer function Bode plot",
    "game theory Nash equilibrium strategy",
    "signal processing Fourier transform frequency",
    "topology manifold surface classification",
    "robotics kinematics joint configuration",
    # 20 adicionales para n=50
    "nuclear physics decay radiation half life",
    "biochemistry enzyme kinetics Michaelis Menten",
    "aerodynamics lift drag airfoil profile",
    "cryptography RSA algorithm encryption",
    "geodesy coordinate system projection map",
    "pharmacology dose response curve",
    "acoustics sound wave interference pattern",
    "numerical methods Newton Raphson iteration",
    "graph theory Euler Hamilton path circuit",
    "semiconductor physics band gap transistor",
    "hydrology watershed drainage basin model",
    "genetics DNA replication transcription",
    "relativity spacetime Lorentz transformation",
    "power systems voltage current transformer",
    "machine learning neural network gradient",
    "geotechnical engineering soil mechanics foundation",
    "information theory entropy channel capacity",
    "biomechanics joint force movement analysis",
    "chemical engineering reactor design kinetics",
    "astrophysics stellar evolution Hertzsprung Russell",
]

# Configuraciones de distractores para Exp 2
DISTRACTOR_CONFIGS = {
    "hard_20pct": {"hard_ratio": 0.20, "label": "20% Hard"},
    "hard_40pct": {"hard_ratio": 0.40, "label": "40% Hard (v2 base)"},
    "hard_60pct": {"hard_ratio": 0.60, "label": "60% Hard"},
    "hard_80pct": {"hard_ratio": 0.80, "label": "80% Hard"},
}

print("=" * 70)
print("V-QMin — STEP 08: TRES EXPERIMENTOS NUEVOS")
print("=" * 70)

# ══════════════════════════════════════════════════════════════
# FUNCIONES COMUNES
# ══════════════════════════════════════════════════════════════

def compute_qtext_for_targets(df):
    """Estimar calidad textual para targets reales."""
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


def build_graph(target_embeddings, n_targets, hard_ratio, seed=42, df_targets=None):
    """
    Construir grafo sintético con ratio de hard distractors variable.
    Si df_targets se provee, Q_text de targets se calcula desde los datos reales
    (mismo método que v2). Si no, usa distribución sintética como fallback.
    Retorna dict con todos los datos del grafo.
    """
    rng = np.random.RandomState(seed)

    distractor_ratio = 4
    n_distractors = n_targets * distractor_ratio
    n_hard = int(n_distractors * hard_ratio)
    n_easy = n_distractors - n_hard
    n_total = n_targets + n_distractors

    # Hard distractors: perturbación gaussiana de targets individuales
    n_hard_t1 = n_hard // 2  # Tier 1: sigma=0.05
    n_hard_t2 = n_hard - n_hard_t1  # Tier 2: sigma=0.10

    hard_emb_parts = []
    if n_hard_t1 > 0:
        base_idx = rng.randint(0, n_targets, size=n_hard_t1)
        noise = rng.randn(n_hard_t1, target_embeddings.shape[1]).astype(np.float32) * 0.05
        t1 = target_embeddings[base_idx] + noise
        t1 = t1 / np.linalg.norm(t1, axis=1, keepdims=True)
        hard_emb_parts.append(t1)

    if n_hard_t2 > 0:
        base_idx = rng.randint(0, n_targets, size=n_hard_t2)
        noise = rng.randn(n_hard_t2, target_embeddings.shape[1]).astype(np.float32) * 0.10
        t2 = target_embeddings[base_idx] + noise
        t2 = t2 / np.linalg.norm(t2, axis=1, keepdims=True)
        hard_emb_parts.append(t2)

    hard_emb = np.concatenate(hard_emb_parts, axis=0) if hard_emb_parts else np.empty((0, target_embeddings.shape[1]))

    # Easy distractors: aleatorios
    centroid = target_embeddings.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    easy_emb = rng.randn(n_easy, target_embeddings.shape[1]).astype(np.float32)
    easy_emb = easy_emb * 0.9 + centroid * 0.1
    easy_emb = easy_emb / np.linalg.norm(easy_emb, axis=1, keepdims=True)

    # Q_text — usar datos reales del DataFrame
    if df_targets is not None:
        np.random.seed(seed)  # Fijar seed para ruido de compute_qtext
        qtext_targets = compute_qtext_for_targets(df_targets)
    else:
        qtext_targets = np.clip(rng.normal(0.77, 0.10, n_targets), 0.10, 0.95)
    qtext_hard = np.clip(rng.normal(0.70, 0.08, n_hard), 0.40, 0.92)
    qtext_easy = np.clip(rng.normal(0.20, 0.10, n_easy), 0.05, 0.40)

    # Ensamblar
    node_types = ['target'] * n_targets + ['easy'] * n_easy + ['hard'] * n_hard
    qtext_all = np.concatenate([qtext_targets, qtext_easy, qtext_hard])
    all_embeddings = np.concatenate([target_embeddings, easy_emb, hard_emb], axis=0)

    # Adyacencia: cada nodo elige 3-8 vecinos
    # con probabilidades categóricas [target, hard, easy]
    idx_target = list(range(0, n_targets))
    idx_easy = list(range(n_targets, n_targets + n_easy))
    idx_hard = list(range(n_targets + n_easy, n_total))
    type_indices = {'target': idx_target, 'easy': idx_easy, 'hard': idx_hard}

    MIN_DEGREE = 3
    MAX_DEGREE = 8
    # Probabilidades categóricas: dado un nodo de tipo X,
    # cada arista va a [target, hard, easy] con estas probabilidades
    p_edges = {
        'target': [0.40, 0.30, 0.30],
        'hard':   [0.35, 0.35, 0.30],
        'easy':   [0.15, 0.25, 0.60],
    }
    dest_order = ['target', 'hard', 'easy']

    adjacency = {i: set() for i in range(n_total)}
    for i in range(n_total):
        src_type = node_types[i]
        degree = rng.randint(MIN_DEGREE, MAX_DEGREE + 1)
        probs = p_edges[src_type]

        for _ in range(degree):
            r = rng.random()
            if r < probs[0]:
                dest_type = 'target'
            elif r < probs[0] + probs[1]:
                dest_type = 'hard'
            else:
                dest_type = 'easy'

            pool = type_indices[dest_type]
            if len(pool) > 0:
                dest = pool[rng.randint(0, len(pool))]
                if dest != i:
                    adjacency[i].add(dest)
                    adjacency[dest].add(i)

    # Semillas
    seeds = []
    for pool in [idx_target, idx_hard, idx_easy]:
        seeds.extend(rng.choice(pool, size=4, replace=False).tolist())

    # Verificar alcanzabilidad
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

    n_edges = sum(len(v) for v in adjacency.values()) // 2

    return {
        'n_total': n_total,
        'n_targets': n_targets,
        'n_easy': n_easy,
        'n_hard': n_hard,
        'node_types': node_types,
        'qtext': qtext_all,
        'embeddings': all_embeddings,
        'adjacency': adjacency,
        'seeds': seeds,
        'n_edges': n_edges,
        'reachability': len(visited) / n_total,
    }


def crawl_bfs(adj, seeds, max_steps):
    """BFS crawler."""
    visited = []
    visited_set = set()
    queue = deque(seeds)
    for s in seeds:
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


def crawl_scored(adj, embeddings, qtext, seeds, query_emb, alpha, max_steps):
    """Best-first crawler con política QMin."""
    visited = []
    visited_set = set()
    frontier = []
    frontier_scores = {}

    for s in seeds:
        cos_sim = float(np.dot(embeddings[s], query_emb))
        rel_visual = (cos_sim + 1.0) / 2.0
        score = alpha * qtext[s] + (1.0 - alpha) * rel_visual
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
                cos_sim = float(np.dot(embeddings[nb], query_emb))
                rel_visual = (cos_sim + 1.0) / 2.0
                nb_score = alpha * qtext[nb] + (1.0 - alpha) * rel_visual

                if nb in frontier_scores:
                    if nb_score < frontier_scores[nb]:
                        frontier_scores[nb] = nb_score
                        heapq.heappush(frontier, (-nb_score, nb))
                else:
                    frontier_scores[nb] = nb_score
                    heapq.heappush(frontier, (-nb_score, nb))

    return visited[:max_steps]


def harvest_rate_at_k(visited, target_set, k):
    top_k = visited[:k]
    hits = sum(1 for n in top_k if n in target_set)
    return hits / k if k > 0 else 0.0


def ndcg_at_k(visited, target_set, k):
    dcg = sum(1.0 / np.log2(i + 2) for i, n in enumerate(visited[:k]) if n in target_set)
    n_rel = min(k, len(target_set))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg > 0 else 0.0


def run_evaluation(graph, query_embeddings, queries, alpha_main=0.2):
    """
    Ejecutar evaluación completa sobre un grafo con N queries.
    Retorna resultados para BFS, Text-Only y V-QMin α=alpha_main.
    """
    adj = graph['adjacency']
    emb = graph['embeddings']
    qt = graph['qtext']
    seeds = graph['seeds']
    target_set = set(range(graph['n_targets']))

    strategies = [
        ('BFS', None),
        ('Text-Only', 1.0),
        (f'V-QMin a={alpha_main}', alpha_main),
    ]

    results = {}
    for sname, alpha in strategies:
        hr_per_query = {k: [] for k in K_VALUES}
        ndcg_per_query = {k: [] for k in K_VALUES}

        for qi in range(len(queries)):
            qemb = query_embeddings[qi]

            if alpha is None:
                visited = crawl_bfs(adj, seeds, K_MAX)
            else:
                visited = crawl_scored(adj, emb, qt, seeds, qemb, alpha, K_MAX)

            for k in K_VALUES:
                hr_per_query[k].append(harvest_rate_at_k(visited, target_set, k))
                ndcg_per_query[k].append(ndcg_at_k(visited, target_set, k))

        results[sname] = {}
        for k in K_VALUES:
            results[sname][f'HR@{k}'] = {
                'mean': float(np.mean(hr_per_query[k])),
                'std': float(np.std(hr_per_query[k])),
                'values': [float(v) for v in hr_per_query[k]],
            }
            results[sname][f'nDCG@{k}'] = {
                'mean': float(np.mean(ndcg_per_query[k])),
                'std': float(np.std(ndcg_per_query[k])),
                'values': [float(v) for v in ndcg_per_query[k]],
            }

    return results


# ══════════════════════════════════════════════════════════════
# CARGAR DATOS
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("CARGANDO DATOS")
print("=" * 70)

df_targets = pd.read_pickle(os.path.join(OUTPUT_DIR, "targets_10k.pkl"))
vitb_img_emb = np.load(os.path.join(OUTPUT_DIR, "img_embeddings_10k.npy"))
vitb_txt_emb = np.load(os.path.join(OUTPUT_DIR, "txt_embeddings_10k.npy"))

vitl_path = os.path.join(OUTPUT_DIR, "vitl14_img_embeddings_10k.npy")
vitl_txt_path = os.path.join(OUTPUT_DIR, "vitl14_txt_embeddings_10k.npy")
has_vitl = os.path.exists(vitl_path) and os.path.exists(vitl_txt_path)

if has_vitl:
    vitl_img_emb = np.load(vitl_path)
    vitl_txt_emb = np.load(vitl_txt_path)
    print(f"  ViT-B/16: img={vitb_img_emb.shape}, txt={vitb_txt_emb.shape}")
    print(f"  ViT-L/14: img={vitl_img_emb.shape}, txt={vitl_txt_emb.shape}")
else:
    print(f"  ViT-B/16: img={vitb_img_emb.shape}, txt={vitb_txt_emb.shape}")
    print(f"  ⚠ ViT-L/14 NO ENCONTRADO — Exp 1 se saltará")
    print(f"    Ejecutar v3_step1_vitl14_embeddings.py primero")

# Pre-computar query embeddings con CLIP
print(f"\n  Cargando CLIP para queries de texto...")
import torch
import open_clip

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

# ViT-B/16 queries
model_b, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai', device=device)
tokenizer_b = open_clip.get_tokenizer('ViT-B-16')
model_b.eval()

with torch.no_grad():
    tokens = tokenizer_b(QUERIES_50).to(device)
    qf = model_b.encode_text(tokens)
    qf = qf / qf.norm(dim=-1, keepdim=True)
    query_emb_b16 = qf.cpu().numpy()  # (50, 512)
print(f"  Query embeddings ViT-B/16: {query_emb_b16.shape}")

# ViT-L/14 queries
if has_vitl:
    model_l, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai', device=device)
    tokenizer_l = open_clip.get_tokenizer('ViT-L-14')
    model_l.eval()

    with torch.no_grad():
        tokens = tokenizer_l(QUERIES_50).to(device)
        qf = model_l.encode_text(tokens)
        qf = qf / qf.norm(dim=-1, keepdim=True)
        query_emb_l14 = qf.cpu().numpy()  # (50, 768)
    print(f"  Query embeddings ViT-L/14: {query_emb_l14.shape}")

    del model_l
    torch.cuda.empty_cache()

del model_b
torch.cuda.empty_cache()

# ══════════════════════════════════════════════════════════════
# EXPERIMENTO 1: ABLACIÓN CLIP (ViT-B/16 vs ViT-L/14)
# ══════════════════════════════════════════════════════════════
if has_vitl:
    print(f"\n{'=' * 70}")
    print("EXPERIMENTO 1: ABLACIÓN CLIP — ViT-B/16 vs ViT-L/14")
    print("=" * 70)

    # Construir grafos idénticos en estructura pero con embeddings diferentes
    print("  Construyendo grafo con ViT-B/16 embeddings...")
    t0 = time.time()
    graph_b16 = build_graph(vitb_img_emb, N_TARGETS, hard_ratio=0.40, seed=SEED, df_targets=df_targets)
    print(f"    {graph_b16['n_total']:,} nodos, {graph_b16['n_edges']:,} aristas ({time.time()-t0:.1f}s)")

    print("  Construyendo grafo con ViT-L/14 embeddings...")
    t0 = time.time()
    graph_l14 = build_graph(vitl_img_emb, N_TARGETS, hard_ratio=0.40, seed=SEED, df_targets=df_targets)
    print(f"    {graph_l14['n_total']:,} nodos, {graph_l14['n_edges']:,} aristas ({time.time()-t0:.1f}s)")

    # Evaluar ambos con 30 queries
    n_queries_exp1 = 30
    print(f"\n  Evaluando ViT-B/16 ({n_queries_exp1} queries)...")
    t0 = time.time()
    results_b16 = run_evaluation(graph_b16, query_emb_b16[:n_queries_exp1],
                                  QUERIES_50[:n_queries_exp1], alpha_main=0.2)
    print(f"    Completado en {time.time()-t0:.1f}s")

    print(f"  Evaluando ViT-L/14 ({n_queries_exp1} queries)...")
    t0 = time.time()
    results_l14 = run_evaluation(graph_l14, query_emb_l14[:n_queries_exp1],
                                  QUERIES_50[:n_queries_exp1], alpha_main=0.2)
    print(f"    Completado en {time.time()-t0:.1f}s")

    # Alpha sweep para ambos
    print("  Alpha sweep ViT-B/16...")
    alpha_sweep_b16 = {}
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        hr_vals = []
        for qi in range(n_queries_exp1):
            visited = crawl_scored(graph_b16['adjacency'], graph_b16['embeddings'],
                                    graph_b16['qtext'], graph_b16['seeds'],
                                    query_emb_b16[qi], alpha, 500)
            hr_vals.append(harvest_rate_at_k(visited, set(range(N_TARGETS)), 500))
        alpha_sweep_b16[alpha] = {'mean': float(np.mean(hr_vals)), 'std': float(np.std(hr_vals))}

    print("  Alpha sweep ViT-L/14...")
    alpha_sweep_l14 = {}
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        hr_vals = []
        for qi in range(n_queries_exp1):
            visited = crawl_scored(graph_l14['adjacency'], graph_l14['embeddings'],
                                    graph_l14['qtext'], graph_l14['seeds'],
                                    query_emb_l14[qi], alpha, 500)
            hr_vals.append(harvest_rate_at_k(visited, set(range(N_TARGETS)), 500))
        alpha_sweep_l14[alpha] = {'mean': float(np.mean(hr_vals)), 'std': float(np.std(hr_vals))}

    # Comparación
    print(f"\n  {'Métrica':<20} {'ViT-B/16':>10} {'ViT-L/14':>10} {'Diff':>10}")
    print("  " + "-" * 52)
    for k in [100, 500, 1000, 5000]:
        b = results_b16['V-QMin a=0.2'][f'HR@{k}']['mean']
        l = results_l14['V-QMin a=0.2'][f'HR@{k}']['mean']
        diff = l - b
        print(f"  HR@{k:<16} {b:>10.4f} {l:>10.4f} {diff:>+10.4f}")

    # Wilcoxon entre modelos
    print(f"\n  Wilcoxon ViT-L/14 vs ViT-B/16 (V-QMin a=0.2):")
    for k in [500, 1000]:
        vals_b = results_b16['V-QMin a=0.2'][f'HR@{k}']['values']
        vals_l = results_l14['V-QMin a=0.2'][f'HR@{k}']['values']
        try:
            stat, p = stats.wilcoxon(vals_l, vals_b, alternative='greater')
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"    HR@{k}: p={p:.4f} {sig}")
        except Exception:
            print(f"    HR@{k}: no calculable (diffs=0?)")

    # Guardar
    exp1_results = {
        "experiment": "CLIP Ablation",
        "n_queries": n_queries_exp1,
        "n_total": graph_b16['n_total'],
        "vitb16": {
            "model": "ViT-B-16",
            "embed_dim": 512,
            "results": results_b16,
            "alpha_sweep": {str(k): v for k, v in alpha_sweep_b16.items()},
        },
        "vitl14": {
            "model": "ViT-L-14",
            "embed_dim": int(vitl_img_emb.shape[1]),
            "results": results_l14,
            "alpha_sweep": {str(k): v for k, v in alpha_sweep_l14.items()},
        },
    }
    with open(os.path.join(OUTPUT_DIR, "exp1_clip_ablation.json"), 'w') as f:
        json.dump(exp1_results, f, indent=2)
    print(f"\n  ✓ Experimento 1 guardado: v3_exp1_clip_ablation.json")

    del graph_b16, graph_l14
    import gc; gc.collect()

else:
    print("\n  ⚠ Saltando Experimento 1 (ViT-L/14 no disponible)")


# ══════════════════════════════════════════════════════════════
# EXPERIMENTO 2: ROBUSTEZ — VARIAR RATIO HARD DISTRACTORS
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("EXPERIMENTO 2: ROBUSTEZ — VARIAR RATIO HARD DISTRACTORS")
print("=" * 70)

exp2_results = {"experiment": "Distractor Robustness", "configs": {}}

for config_name, config in DISTRACTOR_CONFIGS.items():
    hr = config['hard_ratio']
    label = config['label']
    print(f"\n  --- {label} (hard_ratio={hr}) ---")

    t0 = time.time()
    graph = build_graph(vitb_img_emb, N_TARGETS, hard_ratio=hr, seed=SEED, df_targets=df_targets)
    print(f"    Grafo: {graph['n_total']:,} nodos, {graph['n_hard']:,} hard, "
          f"{graph['n_easy']:,} easy, {graph['n_edges']:,} aristas ({time.time()-t0:.1f}s)")
    print(f"    Alcanzabilidad: {graph['reachability']*100:.1f}%")

    # Evaluar con 50 queries
    target_set = set(range(N_TARGETS))
    adj = graph['adjacency']
    emb = graph['embeddings']
    qt = graph['qtext']
    seeds = graph['seeds']

    config_results = {}

    for n_q in [10, 30, 50]:
        strategies_results = {}
        q_embs = query_emb_b16[:n_q]

        for sname, alpha in [('BFS', None), ('Text-Only', 1.0), ('V-QMin a=0.2', 0.2)]:
            hr_per_k = {k: [] for k in K_VALUES}

            for qi in range(n_q):
                if alpha is None:
                    visited = crawl_bfs(adj, seeds, K_MAX)
                else:
                    visited = crawl_scored(adj, emb, qt, seeds, q_embs[qi], alpha, K_MAX)

                for k in K_VALUES:
                    hr_per_k[k].append(harvest_rate_at_k(visited, target_set, k))

            strategies_results[sname] = {}
            for k in K_VALUES:
                strategies_results[sname][f'HR@{k}'] = {
                    'mean': float(np.mean(hr_per_k[k])),
                    'std': float(np.std(hr_per_k[k])),
                    'values': [float(v) for v in hr_per_k[k]],
                }

        config_results[f'n_queries={n_q}'] = strategies_results

        # Print resumen
        vq = strategies_results['V-QMin a=0.2']['HR@500']['mean']
        to = strategies_results['Text-Only']['HR@500']['mean']
        bfs = strategies_results['BFS']['HR@500']['mean']
        improvement = 100 * (vq - to) / to if to > 0 else 0
        print(f"    n={n_q}: V-QMin HR@500={vq:.4f}, Text-Only={to:.4f}, "
              f"BFS={bfs:.4f} | V-QMin vs Text: {improvement:+.1f}%")

    exp2_results['configs'][config_name] = {
        'label': label,
        'hard_ratio': hr,
        'n_hard': graph['n_hard'],
        'n_easy': graph['n_easy'],
        'n_edges': graph['n_edges'],
        'reachability': graph['reachability'],
        'results': config_results,
    }

    del graph; gc.collect()

with open(os.path.join(OUTPUT_DIR, "exp2_distractor_robustness.json"), 'w') as f:
    json.dump(exp2_results, f, indent=2)
print(f"\n  ✓ Experimento 2 guardado: v3_exp2_distractor_robustness.json")


# ══════════════════════════════════════════════════════════════
# EXPERIMENTO 3: ESCALADO DE QUERIES (n=10, 30, 50)
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("EXPERIMENTO 3: ESCALADO DE QUERIES (n=10, 30, 50)")
print("=" * 70)

# Usar grafo base (hard_ratio=0.40) con ViT-B/16
print("  Construyendo grafo base (40% hard)...")
graph_base = build_graph(vitb_img_emb, N_TARGETS, hard_ratio=0.40, seed=SEED, df_targets=df_targets)
print(f"    {graph_base['n_total']:,} nodos, {graph_base['n_edges']:,} aristas")

target_set = set(range(N_TARGETS))
adj = graph_base['adjacency']
emb = graph_base['embeddings']
qt = graph_base['qtext']
seeds = graph_base['seeds']

exp3_results = {"experiment": "Query Scaling", "n_total": graph_base['n_total']}

for n_q in [10, 30, 50]:
    print(f"\n  --- n={n_q} queries ---")
    q_embs = query_emb_b16[:n_q]

    strategies_results = {}
    for sname, alpha in [('BFS', None), ('Text-Only', 1.0),
                          ('V-QMin a=0.2', 0.2), ('V-QMin a=0.3', 0.3)]:
        hr_per_k = {k: [] for k in K_VALUES}
        ndcg_per_k = {k: [] for k in K_VALUES}

        for qi in range(n_q):
            if alpha is None:
                visited = crawl_bfs(adj, seeds, K_MAX)
            else:
                visited = crawl_scored(adj, emb, qt, seeds, q_embs[qi], alpha, K_MAX)

            for k in K_VALUES:
                hr_per_k[k].append(harvest_rate_at_k(visited, target_set, k))
                ndcg_per_k[k].append(ndcg_at_k(visited, target_set, k))

        strategies_results[sname] = {}
        for k in K_VALUES:
            strategies_results[sname][f'HR@{k}'] = {
                'mean': float(np.mean(hr_per_k[k])),
                'std': float(np.std(hr_per_k[k])),
                'values': [float(v) for v in hr_per_k[k]],
            }
            strategies_results[sname][f'nDCG@{k}'] = {
                'mean': float(np.mean(ndcg_per_k[k])),
                'std': float(np.std(ndcg_per_k[k])),
                'values': [float(v) for v in ndcg_per_k[k]],
            }

    # Wilcoxon tests
    wilcoxon_results = {}
    vq_vals_500 = strategies_results['V-QMin a=0.2']['HR@500']['values']
    for baseline in ['BFS', 'Text-Only']:
        base_vals = strategies_results[baseline]['HR@500']['values']
        try:
            stat, p = stats.wilcoxon(vq_vals_500, base_vals, alternative='greater')
            wilcoxon_results[f'V-QMin vs {baseline}'] = {'p_value': float(p), 'n': n_q}
        except Exception:
            wilcoxon_results[f'V-QMin vs {baseline}'] = {'p_value': None, 'n': n_q}

    exp3_results[f'n_queries={n_q}'] = {
        'strategies': strategies_results,
        'wilcoxon': wilcoxon_results,
    }

    # Print
    vq = strategies_results['V-QMin a=0.2']['HR@500']['mean']
    vq_std = strategies_results['V-QMin a=0.2']['HR@500']['std']
    to = strategies_results['Text-Only']['HR@500']['mean']
    p_bfs = wilcoxon_results['V-QMin vs BFS']['p_value']
    p_text = wilcoxon_results['V-QMin vs Text-Only']['p_value']
    print(f"    V-QMin HR@500: {vq:.4f} ± {vq_std:.4f}")
    print(f"    vs Text-Only: {100*(vq-to)/to:+.1f}% | p={p_text}")
    print(f"    vs BFS: p={p_bfs}")

with open(os.path.join(OUTPUT_DIR, "exp3_query_scaling.json"), 'w') as f:
    json.dump(exp3_results, f, indent=2)
print(f"\n  ✓ Experimento 3 guardado: v3_exp3_query_scaling.json")


# ══════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("V-QMin — STEP 08 COMPLETADO ✓")
print("=" * 70)
print("  Archivos generados:")
if has_vitl:
    print("    v3_exp1_clip_ablation.json")
print("    v3_exp2_distractor_robustness.json")
print("    v3_exp3_query_scaling.json")
print(f"\n  Siguiente: step_09_figures_complementary.py")
print("=" * 70)
