#!/usr/bin/env python3
"""
V-QMin Replicación — Step 06: Figuras del Experimento Principal
===================================================================
Genera las figuras principales del paper.

Entrada:  OUTPUT_DIR/evaluation_results.json, alpha_sweep.json, synthetic_graph_50k.pkl
Salida:   OUTPUT_DIR/figures_main/fig[1-8]_*.png
"""

import os
import json
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════
DATASET_DIR = r"I:\RIFINALV3"
OUTPUT_DIR = os.path.join(DATASET_DIR, "vqmin_outputs")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures_main")
os.makedirs(FIG_DIR, exist_ok=True)

DPI = 300
FIGSIZE_WIDE = (10, 5)
FIGSIZE_SQUARE = (7, 5)

print("=" * 70)
print("V-QMin — STEP 06: FIGURAS PRINCIPALES")
print("=" * 70)

# Cargar datos
with open(os.path.join(OUTPUT_DIR, "evaluation_results.json"), 'r', encoding='utf-8') as f:
    results = json.load(f)
with open(os.path.join(OUTPUT_DIR, "alpha_sweep.json"), 'r', encoding='utf-8') as f:
    alpha_sweep = json.load(f)

strategies = results['strategies']
k_values = results['k_values']
queries = results['queries']
n_total = results['n_total']
n_targets = results['n_targets']

STRAT_KEYS = list(strategies.keys())
print(f"  Estrategias: {STRAT_KEYS}")

COLORS_LIST = ['#999999', '#E69F00', '#0072B2', '#009E73', '#CC79A7']
MARKERS_LIST = ['s', '^', 'o', 'D', 'v']
COLORS = {k: c for k, c in zip(STRAT_KEYS, COLORS_LIST)}
MARKERS = {k: m for k, m in zip(STRAT_KEYS, MARKERS_LIST)}
DISPLAY = {k: k.replace('\u03b1', 'α') for k in STRAT_KEYS}

K_BFS = STRAT_KEYS[0]
K_TEXT = STRAT_KEYS[1]
K_VQMIN_ALL = STRAT_KEYS[2:]

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif', 'axes.grid': True,
    'grid.alpha': 0.3, 'figure.dpi': 150, 'savefig.dpi': DPI, 'savefig.bbox': 'tight',
})

# ══════════════════════════════════════════════════════════════
# FIGURA 1: HR@K
# ══════════════════════════════════════════════════════════════
print("  Fig 1: HR@K...")
fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
for sk in STRAT_KEYS:
    means = [strategies[sk][f'HR@{k}']['mean'] for k in k_values]
    stds = [strategies[sk][f'HR@{k}']['std'] for k in k_values]
    ax.plot(k_values, means, color=COLORS[sk], marker=MARKERS[sk], markersize=6, linewidth=2, label=DISPLAY[sk])
    ax.fill_between(k_values, [m-s for m,s in zip(means,stds)], [m+s for m,s in zip(means,stds)], alpha=0.15, color=COLORS[sk])
ax.set_xlabel('K (páginas visitadas)'); ax.set_ylabel('Harvest Rate @ K')
ax.set_title(f'Harvest Rate vs. Presupuesto de Crawling\n({n_targets:,} targets, {n_total:,} nodos, {len(queries)} queries STEM)')
ax.set_xscale('log'); ax.set_xticks(k_values); ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax.set_ylim(0, 1.05); ax.legend(fontsize=10, loc='lower left')
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "fig1_hr_at_k.png")); plt.close()
print("    ✓")

# ══════════════════════════════════════════════════════════════
# FIGURA 2: nDCG@K
# ══════════════════════════════════════════════════════════════
print("  Fig 2: nDCG@K...")
fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
for sk in STRAT_KEYS:
    means = [strategies[sk][f'nDCG@{k}']['mean'] for k in k_values]
    stds = [strategies[sk][f'nDCG@{k}']['std'] for k in k_values]
    ax.plot(k_values, means, color=COLORS[sk], marker=MARKERS[sk], markersize=6, linewidth=2, label=DISPLAY[sk])
    ax.fill_between(k_values, [m-s for m,s in zip(means,stds)], [m+s for m,s in zip(means,stds)], alpha=0.15, color=COLORS[sk])
ax.set_xlabel('K (páginas visitadas)'); ax.set_ylabel('nDCG @ K')
ax.set_title(f'nDCG vs. Presupuesto de Crawling\n({n_targets:,} targets, {n_total:,} nodos)')
ax.set_xscale('log'); ax.set_xticks(k_values); ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax.set_ylim(0, 1.05); ax.legend(fontsize=10, loc='lower left')
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "fig2_ndcg_at_k.png")); plt.close()
print("    ✓")

# ══════════════════════════════════════════════════════════════
# FIGURA 3: ALPHA SWEEP
# ══════════════════════════════════════════════════════════════
print("  Fig 3: Alpha sweep...")
fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
alphas = sorted([float(a) for a in alpha_sweep.keys()])
hr_means = [alpha_sweep[str(a)]['hr_500_mean'] for a in alphas]
hr_stds = [alpha_sweep[str(a)]['hr_500_std'] for a in alphas]
ax.plot(alphas, hr_means, 'o-', color='#0072B2', linewidth=2.5, markersize=8)
ax.fill_between(alphas, [m-s for m,s in zip(hr_means,hr_stds)], [m+s for m,s in zip(hr_means,hr_stds)], alpha=0.2, color='#0072B2')
ax.axvspan(0.1, 0.3, alpha=0.10, color='green', label='Zona recomendada (α=0.1–0.3)')
ax.annotate('Text-Only\n(α=1.0)', xy=(1.0, hr_means[-1]), xytext=(0.85, hr_means[-1]-0.08), fontsize=9, ha='center', arrowprops=dict(arrowstyle='->', color='gray'))
ax.annotate('Visual-Only\n(α=0.0)', xy=(0.0, hr_means[0]), xytext=(0.15, hr_means[0]+0.02), fontsize=9, ha='center', arrowprops=dict(arrowstyle='->', color='gray'))
ax.set_xlabel('α (peso del componente textual)'); ax.set_ylabel('HR@500')
ax.set_title('Impacto del Parámetro α en Harvest Rate\nScore = α·Q_text + (1−α)·Rel_visual')
ax.set_xlim(-0.05, 1.05); ax.legend(fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "fig3_alpha_sweep.png")); plt.close()
print("    ✓")

# ══════════════════════════════════════════════════════════════
# FIGURA 4: BARRAS HR@500
# ══════════════════════════════════════════════════════════════
print("  Fig 4: Barras HR@500...")
fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
bar_order = STRAT_KEYS
hr500 = [strategies[s]['HR@500']['mean'] for s in bar_order]
hr500_std = [strategies[s]['HR@500']['std'] for s in bar_order]
colors = [COLORS[s] for s in bar_order]
bars = ax.barh(range(len(bar_order)), hr500, xerr=hr500_std, color=colors, edgecolor='white', height=0.6, capsize=4)
ax.set_yticks(range(len(bar_order))); ax.set_yticklabels([DISPLAY[s] for s in bar_order], fontsize=11)
ax.set_xlabel('HR@500'); ax.set_title(f'Comparación de Estrategias — HR@500\n({n_total:,} nodos, {n_targets:,} targets)')
ax.set_xlim(0, 1.1)
for i, (v, s) in enumerate(zip(hr500, hr500_std)):
    ax.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=10, fontweight='bold')
ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "fig4_bars_hr500.png")); plt.close()
print("    ✓")

# ══════════════════════════════════════════════════════════════
# FIGURA 5: HEATMAP POR QUERY
# ══════════════════════════════════════════════════════════════
print("  Fig 5: Heatmap...")
fig, ax = plt.subplots(figsize=(12, 6))
strat_order = STRAT_KEYS
query_labels = [q.split()[0].capitalize() for q in queries]
matrix = np.zeros((len(strat_order), len(queries)))
for i, strat in enumerate(strat_order):
    values = strategies[strat]['HR@500']['values']
    for j in range(len(queries)):
        matrix[i, j] = values[j]
im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0.15, vmax=1.0)
ax.set_xticks(range(len(queries))); ax.set_xticklabels(query_labels, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(strat_order))); ax.set_yticklabels([DISPLAY[s] for s in strat_order], fontsize=10)
for i in range(len(strat_order)):
    for j in range(len(queries)):
        val = matrix[i, j]
        color = 'white' if val < 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8).set_label('HR@500')
ax.set_title(f'Harvest Rate @ 500 por Query y Estrategia')
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "fig5_heatmap_queries.png")); plt.close()
print("    ✓")

# ══════════════════════════════════════════════════════════════
# FIGURA 6: MEJORA RELATIVA
# ══════════════════════════════════════════════════════════════
print("  Fig 6: Mejora relativa...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for sk in K_VQMIN_ALL:
    impr = [100*(strategies[sk][f'HR@{k}']['mean'] - strategies[K_BFS][f'HR@{k}']['mean']) / strategies[K_BFS][f'HR@{k}']['mean'] if strategies[K_BFS][f'HR@{k}']['mean'] > 0 else 0 for k in k_values]
    ax1.plot(k_values, impr, color=COLORS[sk], marker=MARKERS[sk], linewidth=2, markersize=6, label=DISPLAY[sk])
ax1.set_xlabel('K'); ax1.set_ylabel('Mejora relativa (%)'); ax1.set_title('Mejora sobre BFS')
ax1.set_xscale('log'); ax1.set_xticks(k_values); ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

for sk in K_VQMIN_ALL:
    impr = [100*(strategies[sk][f'HR@{k}']['mean'] - strategies[K_TEXT][f'HR@{k}']['mean']) / strategies[K_TEXT][f'HR@{k}']['mean'] if strategies[K_TEXT][f'HR@{k}']['mean'] > 0 else 0 for k in k_values]
    ax2.plot(k_values, impr, color=COLORS[sk], marker=MARKERS[sk], linewidth=2, markersize=6, label=DISPLAY[sk])
ax2.set_xlabel('K'); ax2.set_ylabel('Mejora relativa (%)'); ax2.set_title('Mejora sobre Text-Only')
ax2.set_xscale('log'); ax2.set_xticks(k_values); ax2.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
plt.suptitle('Mejora Relativa de V-QMin sobre Baselines', fontsize=13, y=1.02)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "fig6_relative_improvement.png")); plt.close()
print("    ✓")

# ══════════════════════════════════════════════════════════════
# FIGURA 7: DISTRIBUCIÓN DE SIMILITUDES
# ══════════════════════════════════════════════════════════════
print("  Fig 7: Distribución similitudes...")
with open(os.path.join(OUTPUT_DIR, "synthetic_graph_50k.pkl"), 'rb') as f:
    graph = pickle.load(f)
embs = graph['embeddings']; nt = graph['n_targets']; ne = graph['n_easy']; nh = graph['n_hard']
np.random.seed(42)
ns = 500
t_s = embs[np.random.choice(nt, ns, replace=False)]
e_s = embs[nt + np.random.choice(ne, ns, replace=False)]
h_s = embs[nt + ne + np.random.choice(nh, ns, replace=False)]
cent = t_s.mean(axis=0); cent = cent / np.linalg.norm(cent)

fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
ax.hist(t_s @ cent, bins=50, alpha=0.6, color='#009E73', label=f'Targets (n={nt:,})', density=True)
ax.hist(h_s @ cent, bins=50, alpha=0.6, color='#E69F00', label=f'Hard dist. (n={nh:,})', density=True)
ax.hist(e_s @ cent, bins=50, alpha=0.6, color='#999999', label=f'Easy dist. (n={ne:,})', density=True)
ax.set_xlabel('Similitud coseno con centroide de targets'); ax.set_ylabel('Densidad')
ax.set_title('Distribución de Similitudes por Tipo de Nodo'); ax.legend(fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "fig7_similarity_distribution.png")); plt.close()
print("    ✓")

# ══════════════════════════════════════════════════════════════
# FIGURA 8: TABLA RESUMEN
# ══════════════════════════════════════════════════════════════
print("  Fig 8: Tabla resumen...")
fig, ax = plt.subplots(figsize=(10, 4)); ax.axis('off')
col_labels = ['Estrategia', 'HR@100', 'HR@500', 'HR@1000', 'HR@5000', 'nDCG@500']
table_data = []
for s in STRAT_KEYS:
    row = [DISPLAY[s]]
    for metric in ['HR@100', 'HR@500', 'HR@1000', 'HR@5000', 'nDCG@500']:
        m = strategies[s][metric]['mean']; sd = strategies[s][metric]['std']
        row.append(f'{m:.3f}±{sd:.3f}')
    table_data.append(row)
table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.2, 1.8)
for j in range(len(col_labels)):
    table[0, j].set_facecolor('#2C3E50'); table[0, j].set_text_props(color='white', fontweight='bold')
for i in range(len(STRAT_KEYS)):
    color = '#ECF0F1' if i % 2 == 0 else 'white'
    for j in range(len(col_labels)):
        table[i+1, j].set_facecolor(color)
for j in range(1, len(col_labels)):
    vals = [strategies[STRAT_KEYS[i]][col_labels[j]]['mean'] for i in range(len(STRAT_KEYS))]
    best_i = np.argmax(vals)
    table[best_i+1, j].set_text_props(fontweight='bold', color='#006600')
ax.set_title(f'Resultados Experimentales — V-QMin\n{n_total:,} nodos, {n_targets:,} targets, Wilcoxon p<0.001', fontsize=12, pad=20)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "fig8_results_table.png")); plt.close()
print("    ✓")

# ══════════════════════════════════════════════════════════════
# RESUMEN
# ══════════════════════════════════════════════════════════════
figs = sorted([f for f in os.listdir(FIG_DIR) if f.endswith('.png')])
print(f"\n{'=' * 70}")
print(f"STEP 06 COMPLETADO ✓ — {len(figs)} figuras en {FIG_DIR}")
for f in figs:
    print(f"  {f} ({os.path.getsize(os.path.join(FIG_DIR, f))/1024:.0f} KB)")
print(f"\nSiguiente: step_07_vitl14_embeddings.py")
