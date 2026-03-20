#!/usr/bin/env python3
"""
V-QMin Replicación v4 — Step 08: Tres Experimentos Complementarios
=====================================================================
Experimento 1: Ablación CLIP — ViT-B/16 vs ViT-L/14 (1000 queries)
Experimento 2: Robustez — Variar ratio hard distractors (20%, 40%, 60%, 80%)
Experimento 3: Escalado queries — n=50, 100, 300, 1000

CAMBIO v4: Las 1000 queries se muestrean aleatoriamente del campo 'question'
de df_targets (seed=42), usando el mismo índice guardado en queries_1000.json.
Si este archivo no existe, se regenera con el mismo seed.

Salidas:
  - exp1_clip_ablation.json
  - exp2_distractor_robustness.json
  - exp3_query_scaling.json
"""

import os
import sys
import time
import json
import heapq
import random
import gc
import numpy as np
import pandas as pd
from collections import deque
from scipy import stats

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════
DATASET_DIR = r"I:\RIFINALV4"
OUTPUT_DIR  = os.path.join(DATASET_DIR, "vqmin_outputs")
SEED        = 42
random.seed(SEED)
np.random.seed(SEED)

N_TARGETS = 10000
N_QUERIES = 1000       # v4: 1000 queries del dataset
K_VALUES  = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
K_MAX     = max(K_VALUES)

# Configuraciones de distractores para Exp 2
DISTRACTOR_CONFIGS = {
    "hard_20pct": {"hard_ratio": 0.20, "label": "20% Hard"},
    "hard_40pct": {"hard_ratio": 0.40, "label": "40% Hard (base)"},
    "hard_60pct": {"hard_ratio": 0.60, "label": "60% Hard"},
    "hard_80pct": {"hard_ratio": 0.80, "label": "80% Hard"},
}

print("=" * 70)
print("V-QMin v4 — STEP 08: TRES EXPERIMENTOS COMPLEMENTARIOS")
print(f"  N_QUERIES = {N_QUERIES} (dataset, seed={SEED})")
print("=" * 70)

# ══════════════════════════════════════════════════════════════
# FUNCIONES COMUNES (iguales a v3)
# ══════════════════════════════════════════════════════════════

def compute_qtext_for_targets(df):
    n    = len(df)
    base = np.full(n, 0.50)
    ans_len = df['answer'].str.len().values
    base += np.where(ans_len > 1000, 0.20, ans_len / 1000 * 0.20)
    q_len = df['question'].str.len().values
    base += np.where(q_len > 200, 0.10, q_len / 200 * 0.10)
    diff  = df['difficulty'].fillna(2).values.astype(float)
    base += (diff - 1) * 0.02
    noise = np.random.normal(0, 0.03, n)
    return np.clip(base + noise, 0.10, 0.95)


def build_graph(target_embeddings, n_targets, hard_ratio, seed=42, df_targets=None):
    rng           = np.random.RandomState(seed)
    n_distractors = n_targets * 4
    n_hard        = int(n_distractors * hard_ratio)
    n_easy        = n_distractors - n_hard
    n_total       = n_targets + n_distractors

    n_hard_t1 = n_hard // 2
    n_hard_t2 = n_hard - n_hard_t1

    hard_parts = []
    if n_hard_t1 > 0:
        idx   = rng.randint(0, n_targets, n_hard_t1)
        noise = rng.randn(n_hard_t1, target_embeddings.shape[1]).astype(np.float32) * 0.05
        t1    = target_embeddings[idx] + noise
        t1   /= np.linalg.norm(t1, axis=1, keepdims=True)
        hard_parts.append(t1)
    if n_hard_t2 > 0:
        idx   = rng.randint(0, n_targets, n_hard_t2)
        noise = rng.randn(n_hard_t2, target_embeddings.shape[1]).astype(np.float32) * 0.10
        t2    = target_embeddings[idx] + noise
        t2   /= np.linalg.norm(t2, axis=1, keepdims=True)
        hard_parts.append(t2)

    hard_emb  = np.concatenate(hard_parts) if hard_parts else np.empty((0, target_embeddings.shape[1]))
    centroid  = target_embeddings.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    easy_emb  = rng.randn(n_easy, target_embeddings.shape[1]).astype(np.float32) * 0.9 + centroid * 0.1
    easy_emb /= np.linalg.norm(easy_emb, axis=1, keepdims=True)

    if df_targets is not None:
        np.random.seed(seed)
        qtext_targets = compute_qtext_for_targets(df_targets)
    else:
        qtext_targets = np.clip(rng.normal(0.77, 0.10, n_targets), 0.10, 0.95)

    qtext_hard  = np.clip(rng.normal(0.70, 0.08, n_hard), 0.40, 0.92)
    qtext_easy  = np.clip(rng.normal(0.20, 0.10, n_easy), 0.05, 0.40)
    node_types  = ['target'] * n_targets + ['easy'] * n_easy + ['hard'] * n_hard
    qtext_all   = np.concatenate([qtext_targets, qtext_easy, qtext_hard])
    all_emb     = np.concatenate([target_embeddings, easy_emb, hard_emb], axis=0)

    idx_target  = list(range(0, n_targets))
    idx_easy    = list(range(n_targets, n_targets + n_easy))
    idx_hard    = list(range(n_targets + n_easy, n_total))
    type_indices = {'target': idx_target, 'easy': idx_easy, 'hard': idx_hard}

    p_edges = {
        'target': [0.40, 0.30, 0.30],
        'hard':   [0.35, 0.35, 0.30],
        'easy':   [0.15, 0.25, 0.60],
    }
    dest_order = ['target', 'hard', 'easy']
    adjacency  = {i: set() for i in range(n_total)}

    for i in range(n_total):
        src_type = node_types[i]
        degree   = rng.randint(3, 9)
        probs    = p_edges[src_type]
        for _ in range(degree):
            r = rng.random()
            if r < probs[0]:             dt = 'target'
            elif r < probs[0]+probs[1]:  dt = 'hard'
            else:                         dt = 'easy'
            pool = type_indices[dt]
            if pool:
                dest = pool[rng.randint(0, len(pool))]
                if dest != i:
                    adjacency[i].add(dest)
                    adjacency[dest].add(i)

    seeds = []
    for pool in [idx_target, idx_hard, idx_easy]:
        seeds.extend(rng.choice(pool, size=4, replace=False).tolist())

    visited = set(seeds)
    queue   = deque(seeds)
    while queue:
        node = queue.popleft()
        for nb in adjacency[node]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)

    n_edges = sum(len(v) for v in adjacency.values()) // 2
    return {
        'n_total': n_total, 'n_targets': n_targets,
        'n_easy': n_easy,   'n_hard': n_hard,
        'node_types': node_types, 'qtext': qtext_all,
        'embeddings': all_emb, 'adjacency': adjacency,
        'seeds': seeds, 'n_edges': n_edges,
        'reachability': len(visited) / n_total,
    }


def crawl_bfs(adj, seeds, max_steps):
    visited, vs = [], set()
    queue = deque(seeds)
    for s in seeds: vs.add(s)
    while queue and len(visited) < max_steps:
        node = queue.popleft()
        visited.append(node)
        nbs = list(adj.get(node, []))
        random.shuffle(nbs)
        for nb in nbs:
            if nb not in vs:
                vs.add(nb); queue.append(nb)
    return visited[:max_steps]


def crawl_scored(adj, embeddings, qtext, seeds, query_emb, alpha, max_steps):
    visited, vs = [], set()
    frontier, fs = [], {}
    for s in seeds:
        sc = alpha * qtext[s] + (1.0 - alpha) * (float(np.dot(embeddings[s], query_emb)) + 1.0) / 2.0
        heapq.heappush(frontier, (-sc, s)); fs[s] = sc
    while frontier and len(visited) < max_steps:
        _, node = heapq.heappop(frontier)
        if node in vs: continue
        visited.append(node); vs.add(node)
        for nb in adj.get(node, []):
            if nb not in vs:
                sc = alpha * qtext[nb] + (1.0 - alpha) * (float(np.dot(embeddings[nb], query_emb)) + 1.0) / 2.0
                if nb not in fs or sc < fs[nb]:
                    fs[nb] = sc; heapq.heappush(frontier, (-sc, nb))
    return visited[:max_steps]


def hr_at_k(visited, ts, k):
    return sum(1 for n in visited[:k] if n in ts) / k if k > 0 else 0.0


def ndcg_at_k(visited, ts, k):
    dcg  = sum(1.0 / np.log2(i+2) for i, n in enumerate(visited[:k]) if n in ts)
    idcg = sum(1.0 / np.log2(i+2) for i in range(min(k, len(ts))))
    return dcg / idcg if idcg > 0 else 0.0


def run_evaluation(graph, query_embeddings_arr, n_q, alpha_main=0.2, label=""):
    adj  = graph['adjacency']; emb = graph['embeddings']
    qt   = graph['qtext'];     seeds = graph['seeds']
    ts   = set(range(graph['n_targets']))
    strats = [('BFS', None), ('Text-Only', 1.0), (f'V-QMin a={alpha_main}', alpha_main)]
    results = {}
    for sname, alpha in strats:
        hr_pk   = {k: [] for k in K_VALUES}
        ndcg_pk = {k: [] for k in K_VALUES}
        for qi in range(n_q):
            qemb    = query_embeddings_arr[qi]
            visited = crawl_bfs(adj, seeds, K_MAX) if alpha is None \
                      else crawl_scored(adj, emb, qt, seeds, qemb, alpha, K_MAX)
            for k in K_VALUES:
                hr_pk[k].append(hr_at_k(visited, ts, k))
                ndcg_pk[k].append(ndcg_at_k(visited, ts, k))
        results[sname] = {}
        for k in K_VALUES:
            results[sname][f'HR@{k}']   = {'mean': float(np.mean(hr_pk[k])),
                                            'std':  float(np.std(hr_pk[k])),
                                            'values': [float(v) for v in hr_pk[k]]}
            results[sname][f'nDCG@{k}'] = {'mean': float(np.mean(ndcg_pk[k])),
                                            'std':  float(np.std(ndcg_pk[k])),
                                            'values': [float(v) for v in ndcg_pk[k]]}
    return results


# ══════════════════════════════════════════════════════════════
# CARGAR DATOS
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("CARGANDO DATOS")
print("=" * 70)

df_targets   = pd.read_pickle(os.path.join(OUTPUT_DIR, "targets_10k.pkl"))
vitb_img_emb = np.load(os.path.join(OUTPUT_DIR, "img_embeddings_10k.npy"))
vitb_txt_emb = np.load(os.path.join(OUTPUT_DIR, "txt_embeddings_10k.npy"))

vitl_path     = os.path.join(OUTPUT_DIR, "vitl14_img_embeddings_10k.npy")
vitl_txt_path = os.path.join(OUTPUT_DIR, "vitl14_txt_embeddings_10k.npy")
has_vitl      = os.path.exists(vitl_path) and os.path.exists(vitl_txt_path)

if has_vitl:
    vitl_img_emb = np.load(vitl_path)
    vitl_txt_emb = np.load(vitl_txt_path)
    print(f"  ViT-B/16: img={vitb_img_emb.shape}")
    print(f"  ViT-L/14: img={vitl_img_emb.shape}")
else:
    print(f"  ViT-B/16: img={vitb_img_emb.shape}")
    print(f"  ⚠ ViT-L/14 no encontrado — Exp 1 se saltará")

# ══════════════════════════════════════════════════════════════
# CARGAR / REGENERAR LAS 1000 QUERIES
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"QUERIES — {N_QUERIES} del dataset")
print("=" * 70)

queries_json = os.path.join(OUTPUT_DIR, "queries_1000.json")
if os.path.exists(queries_json):
    with open(queries_json, 'r', encoding='utf-8') as f:
        qdata = json.load(f)
    QUERIES    = qdata['queries']
    query_idxs = qdata['indices']
    print(f"  Cargadas desde queries_1000.json ({len(QUERIES)} queries)")
else:
    print(f"  queries_1000.json no encontrado — regenerando con seed={SEED}")
    rng        = np.random.RandomState(SEED)
    query_idxs = rng.choice(len(df_targets), size=N_QUERIES, replace=False).tolist()
    QUERIES    = df_targets['question'].iloc[query_idxs].tolist()
    with open(queries_json, 'w', encoding='utf-8') as f:
        json.dump({'n_queries': N_QUERIES, 'seed': SEED,
                   'indices': query_idxs, 'queries': QUERIES}, f,
                  indent=2, ensure_ascii=False)
    print(f"  Generadas y guardadas: {len(QUERIES)} queries")

print(f"  Ejemplo [0]: {QUERIES[0][:80]}...")

# ══════════════════════════════════════════════════════════════
# CLIP EMBEDDINGS DE LAS 1000 QUERIES
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("CLIP EMBEDDINGS — QUERIES")
print("=" * 70)

import torch, open_clip

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

CLIP_BATCH = 256

# ViT-B/16
model_b, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai', device=device)
tok_b = open_clip.get_tokenizer('ViT-B-16'); model_b.eval()
all_qemb = []
with torch.no_grad():
    for bs in range(0, len(QUERIES), CLIP_BATCH):
        be = min(bs + CLIP_BATCH, len(QUERIES))
        f  = model_b.encode_text(tok_b(QUERIES[bs:be]).to(device))
        f  = f / f.norm(dim=-1, keepdim=True)
        all_qemb.append(f.cpu().numpy())
query_emb_b16 = np.concatenate(all_qemb)
del model_b; torch.cuda.empty_cache()
print(f"  ViT-B/16 query embeddings: {query_emb_b16.shape}")

# ViT-L/14
if has_vitl:
    model_l, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai', device=device)
    tok_l = open_clip.get_tokenizer('ViT-L-14'); model_l.eval()
    all_qemb_l = []
    with torch.no_grad():
        for bs in range(0, len(QUERIES), CLIP_BATCH):
            be = min(bs + CLIP_BATCH, len(QUERIES))
            f  = model_l.encode_text(tok_l(QUERIES[bs:be]).to(device))
            f  = f / f.norm(dim=-1, keepdim=True)
            all_qemb_l.append(f.cpu().numpy())
    query_emb_l14 = np.concatenate(all_qemb_l)
    del model_l; torch.cuda.empty_cache()
    print(f"  ViT-L/14 query embeddings: {query_emb_l14.shape}")

# ══════════════════════════════════════════════════════════════
# EXPERIMENTO 1: ABLACIÓN CLIP (ViT-B/16 vs ViT-L/14)
# ══════════════════════════════════════════════════════════════
if has_vitl:
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENTO 1: ABLACIÓN CLIP — {N_QUERIES} queries")
    print("=" * 70)

    print("  Construyendo grafo ViT-B/16...")
    graph_b16 = build_graph(vitb_img_emb, N_TARGETS, hard_ratio=0.40,
                            seed=SEED, df_targets=df_targets)
    print(f"    {graph_b16['n_total']:,} nodos, {graph_b16['n_edges']:,} aristas")

    print("  Construyendo grafo ViT-L/14...")
    graph_l14 = build_graph(vitl_img_emb, N_TARGETS, hard_ratio=0.40,
                            seed=SEED, df_targets=df_targets)

    t0 = time.time()
    print(f"  Evaluando ViT-B/16 ({N_QUERIES} queries)...")
    results_b16 = run_evaluation(graph_b16, query_emb_b16, N_QUERIES, alpha_main=0.2)
    print(f"    ✓ {(time.time()-t0)/60:.1f} min")

    t0 = time.time()
    print(f"  Evaluando ViT-L/14 ({N_QUERIES} queries)...")
    results_l14 = run_evaluation(graph_l14, query_emb_l14, N_QUERIES, alpha_main=0.2)
    print(f"    ✓ {(time.time()-t0)/60:.1f} min")

    # Alpha sweep para ambos modelos
    alpha_sweep_b16, alpha_sweep_l14 = {}, {}
    ts_b16 = set(range(graph_b16['n_targets']))
    ts_l14 = set(range(graph_l14['n_targets']))

    for alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]:
        vals_b, vals_l = [], []
        for qi in range(N_QUERIES):
            v_b = crawl_scored(graph_b16['adjacency'], graph_b16['embeddings'],
                               graph_b16['qtext'], graph_b16['seeds'],
                               query_emb_b16[qi], alpha, 500)
            v_l = crawl_scored(graph_l14['adjacency'], graph_l14['embeddings'],
                               graph_l14['qtext'], graph_l14['seeds'],
                               query_emb_l14[qi], alpha, 500)
            vals_b.append(hr_at_k(v_b, ts_b16, 500))
            vals_l.append(hr_at_k(v_l, ts_l14, 500))
        alpha_sweep_b16[alpha] = {'mean': float(np.mean(vals_b)), 'std': float(np.std(vals_b))}
        alpha_sweep_l14[alpha] = {'mean': float(np.mean(vals_l)), 'std': float(np.std(vals_l))}
        print(f"  α={alpha:.1f}: B16={np.mean(vals_b):.4f}  L14={np.mean(vals_l):.4f}")

    exp1_results = {
        "experiment": "CLIP Ablation ViT-B/16 vs ViT-L/14",
        "n_queries": N_QUERIES,
        "vitb16": {"results": results_b16, "alpha_sweep": alpha_sweep_b16},
        "vitl14": {"results": results_l14, "alpha_sweep": alpha_sweep_l14},
    }
    with open(os.path.join(OUTPUT_DIR, "exp1_clip_ablation.json"), 'w') as f:
        json.dump(exp1_results, f, indent=2)
    print(f"  ✓ exp1_clip_ablation.json guardado")
    del graph_b16, graph_l14; gc.collect()

else:
    print("\n  EXPERIMENTO 1: Saltado (ViT-L/14 no disponible)")

# ══════════════════════════════════════════════════════════════
# EXPERIMENTO 2: ROBUSTEZ — Variar ratio hard distractors
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"EXPERIMENTO 2: ROBUSTEZ DISTRACTORS — {N_QUERIES} queries")
print("=" * 70)

exp2_results = {"experiment": "Distractor Robustness", "n_queries": N_QUERIES, "configs": {}}

for config_name, config in DISTRACTOR_CONFIGS.items():
    hr    = config['hard_ratio']
    label = config['label']
    print(f"\n  --- {label} ---")
    t0 = time.time()
    graph = build_graph(vitb_img_emb, N_TARGETS, hard_ratio=hr,
                        seed=SEED, df_targets=df_targets)
    print(f"    {graph['n_total']:,} nodos | reach={graph['reachability']*100:.1f}% ({time.time()-t0:.1f}s)")

    ts   = set(range(N_TARGETS))
    adj  = graph['adjacency']; emb = graph['embeddings']
    qt   = graph['qtext'];     gseeds = graph['seeds']

    config_results = {}
    for n_q in [50, 100, 300, 1000]:
        strats_r = {}
        q_embs   = query_emb_b16[:n_q]
        for sname, alpha in [('BFS', None), ('Text-Only', 1.0), ('V-QMin a=0.2', 0.2)]:
            hr_pk = {k: [] for k in K_VALUES}
            for qi in range(n_q):
                vis = crawl_bfs(adj, gseeds, K_MAX) if alpha is None \
                      else crawl_scored(adj, emb, qt, gseeds, q_embs[qi], alpha, K_MAX)
                for k in K_VALUES:
                    hr_pk[k].append(hr_at_k(vis, ts, k))
            strats_r[sname] = {f'HR@{k}': {'mean': float(np.mean(hr_pk[k])),
                                             'std':  float(np.std(hr_pk[k])),
                                             'values': [float(v) for v in hr_pk[k]]}
                                for k in K_VALUES}
        config_results[f'n_queries={n_q}'] = strats_r
        vq  = strats_r['V-QMin a=0.2']['HR@500']['mean']
        to  = strats_r['Text-Only']['HR@500']['mean']
        bfs = strats_r['BFS']['HR@500']['mean']
        print(f"    n={n_q}: V-QMin={vq:.4f}  Text={to:.4f}  BFS={bfs:.4f}  "
              f"(vs Text: {100*(vq-to)/to:+.1f}%)")

    exp2_results['configs'][config_name] = {
        'label': label, 'hard_ratio': hr,
        'n_hard': graph['n_hard'], 'n_easy': graph['n_easy'],
        'n_edges': graph['n_edges'], 'reachability': graph['reachability'],
        'results': config_results,
    }
    del graph; gc.collect()

with open(os.path.join(OUTPUT_DIR, "exp2_distractor_robustness.json"), 'w') as f:
    json.dump(exp2_results, f, indent=2)
print(f"\n  ✓ exp2_distractor_robustness.json guardado")

# ══════════════════════════════════════════════════════════════
# EXPERIMENTO 3: ESCALADO DE QUERIES — n=50, 100, 300, 1000
# ══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("EXPERIMENTO 3: ESCALADO DE QUERIES — n=50, 100, 300, 1000")
print("=" * 70)

print("  Construyendo grafo base (40% hard)...")
graph_base = build_graph(vitb_img_emb, N_TARGETS, hard_ratio=0.40,
                         seed=SEED, df_targets=df_targets)
print(f"    {graph_base['n_total']:,} nodos, {graph_base['n_edges']:,} aristas")

ts_base = set(range(N_TARGETS))
adj_b   = graph_base['adjacency']; emb_b = graph_base['embeddings']
qt_b    = graph_base['qtext'];     seeds_b = graph_base['seeds']

exp3_results = {"experiment": "Query Scaling", "n_total": graph_base['n_total']}

for n_q in [50, 100, 300, 1000]:
    print(f"\n  --- n={n_q} queries ---")
    t0     = time.time()
    q_embs = query_emb_b16[:n_q]
    strats_r = {}

    for sname, alpha in [('BFS', None), ('Text-Only', 1.0),
                          ('V-QMin a=0.2', 0.2), ('V-QMin a=0.3', 0.3)]:
        hr_pk   = {k: [] for k in K_VALUES}
        ndcg_pk = {k: [] for k in K_VALUES}
        for qi in range(n_q):
            vis = crawl_bfs(adj_b, seeds_b, K_MAX) if alpha is None \
                  else crawl_scored(adj_b, emb_b, qt_b, seeds_b, q_embs[qi], alpha, K_MAX)
            for k in K_VALUES:
                hr_pk[k].append(hr_at_k(vis, ts_base, k))
                ndcg_pk[k].append(ndcg_at_k(vis, ts_base, k))
        strats_r[sname] = {}
        for k in K_VALUES:
            strats_r[sname][f'HR@{k}']   = {'mean': float(np.mean(hr_pk[k])),
                                              'std':  float(np.std(hr_pk[k])),
                                              'values': [float(v) for v in hr_pk[k]]}
            strats_r[sname][f'nDCG@{k}'] = {'mean': float(np.mean(ndcg_pk[k])),
                                              'std':  float(np.std(ndcg_pk[k])),
                                              'values': [float(v) for v in ndcg_pk[k]]}

    # Wilcoxon
    wilcoxon_r = {}
    vq_vals = strats_r['V-QMin a=0.2']['HR@500']['values']
    for baseline in ['BFS', 'Text-Only']:
        base_vals = strats_r[baseline]['HR@500']['values']
        try:
            _, p = stats.wilcoxon(vq_vals, base_vals, alternative='greater')
        except Exception:
            p = None
        wilcoxon_r[f'V-QMin vs {baseline}'] = {'p_value': float(p) if p else None, 'n': n_q}

    exp3_results[f'n_queries={n_q}'] = {'strategies': strats_r, 'wilcoxon': wilcoxon_r}

    vq  = strats_r['V-QMin a=0.2']['HR@500']['mean']
    to  = strats_r['Text-Only']['HR@500']['mean']
    p_w = wilcoxon_r['V-QMin vs Text-Only']['p_value']
    sig = "***" if p_w and p_w < 0.001 else "**" if p_w and p_w < 0.01 else "*" if p_w and p_w < 0.05 else "ns"
    print(f"    V-QMin HR@500={vq:.4f}  Text-Only={to:.4f}  "
          f"Wilcoxon p={p_w:.2e} {sig}  ({(time.time()-t0)/60:.1f}min)")

with open(os.path.join(OUTPUT_DIR, "exp3_query_scaling.json"), 'w') as f:
    json.dump(exp3_results, f, indent=2)
print(f"\n  ✓ exp3_query_scaling.json guardado")

del graph_base; gc.collect()

print(f"\n{'=' * 70}")
print("STEP 08 COMPLETADO ✓ — Siguiente: step_09_figures_complementary.py")
print("=" * 70)
