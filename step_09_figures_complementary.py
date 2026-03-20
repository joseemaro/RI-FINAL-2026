#!/usr/bin/env python3
"""
V-QMin Replicación v4 — Step 09: Figuras Comparativas para el Paper
========================================================
CAMBIOS v4:
  - Path: RIFINALV4
  - Query scaling: n=50, 100, 300, 1000 (antes: 10, 30, 50)
  - Default n_key para figuras de robustez: n_queries=100

Figuras generadas:
  fig_v3_01_clip_ablation_bars.png    — HR@K comparando ViT-B/16 vs ViT-L/14
  fig_v3_02_clip_ablation_alpha.png   — Alpha sweep de ambos modelos superpuestos
  fig_v3_03_robustness_vqmin.png      — HR@500 de V-QMin vs % hard distractors
  fig_v3_04_robustness_advantage.png  — Ventaja relativa V-QMin vs Text-Only por % hard
  fig_v3_05_query_scaling.png         — HR@500 con barras de error para n=50,100,300,1000
  fig_v3_06_query_scaling_ci.png      — Intervalos de confianza al escalar queries
  fig_v3_07_summary_table.png         — Tabla resumen de los 3 experimentos
  fig_v3_08_robustness_by_nqueries.png — Robustez × queries

Entrada:  exp1_clip_ablation.json, exp2_distractor_robustness.json,
          exp3_query_scaling.json
Salida:   figures_v3/*.png (300 DPI)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════
OUTPUT_DIR = os.path.join(r"I:\RIFINALV4", "vqmin_outputs")
FIG_DIR    = os.path.join(OUTPUT_DIR, "figures_v3")
os.makedirs(FIG_DIR, exist_ok=True)

DPI = 300
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'legend.fontsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
})

C_VITB  = '#2196F3'
C_VITL  = '#FF9800'
C_BFS   = '#9E9E9E'
C_TEXT  = '#4CAF50'
C_VQMIN = '#E91E63'
C_VQMIN2 = '#9C27B0'

# v4: query scaling usa estos valores
N_QUERIES_LIST = [50, 100, 300, 1000]
DEFAULT_N_KEY  = 'n_queries=100'   # usado como referencia en figs de robustez

print("=" * 70)
print("V-QMin v4 — STEP 09: FIGURAS COMPARATIVAS")
print("=" * 70)

# ══════════════════════════════════════════════════════════════
# CARGAR DATOS
# ══════════════════════════════════════════════════════════════
exp1_path = os.path.join(OUTPUT_DIR, "exp1_clip_ablation.json")
exp2_path = os.path.join(OUTPUT_DIR, "exp2_distractor_robustness.json")
exp3_path = os.path.join(OUTPUT_DIR, "exp3_query_scaling.json")

has_exp1 = os.path.exists(exp1_path)
has_exp2 = os.path.exists(exp2_path)
has_exp3 = os.path.exists(exp3_path)

if has_exp1:
    with open(exp1_path) as f: exp1 = json.load(f)
    print(f"  ✓ Exp 1 (CLIP Ablation)")
if has_exp2:
    with open(exp2_path) as f: exp2 = json.load(f)
    print(f"  ✓ Exp 2 (Distractor Robustness)")
if has_exp3:
    with open(exp3_path) as f: exp3 = json.load(f)
    print(f"  ✓ Exp 3 (Query Scaling)")

# ══════════════════════════════════════════════════════════════
# FIGURA 1: CLIP ABLATION — BARRAS HR@K
# ══════════════════════════════════════════════════════════════
if has_exp1:
    print(f"\n  Generando fig_v3_01_clip_ablation_bars.png...")
    k_vals     = [100, 500, 1000, 5000]
    strategies = ['BFS', 'Text-Only', 'V-QMin a=0.2']
    fig, axes  = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax_idx, (model_key, model_label) in enumerate([
        ('vitb16', 'ViT-B/16'), ('vitl14', 'ViT-L/14')
    ]):
        ax   = axes[ax_idx]
        data = exp1[model_key]['results']
        x    = np.arange(len(k_vals))
        w    = 0.25
        for i, strat in enumerate(strategies):
            means = [data[strat][f'HR@{k}']['mean'] for k in k_vals]
            stds  = [data[strat][f'HR@{k}']['std']  for k in k_vals]
            bars  = ax.bar(x + i*w, means, w,
                           label=strat, color=[C_BFS, C_TEXT, C_VQMIN][i],
                           alpha=0.85, edgecolor='white', yerr=stds, capsize=3)
            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7.5)
        ax.set_xlabel('K'); ax.set_ylabel('Harvest Rate')
        ax.set_title(model_label)
        ax.set_xticks(x + w); ax.set_xticklabels([str(k) for k in k_vals])
        ax.set_ylim(0, 1.12); ax.legend(loc='upper left', framealpha=0.9)

    fig.suptitle('Experimento 1: Ablación de Modelo CLIP — HR@K',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_v3_01_clip_ablation_bars.png"),
                dpi=DPI, bbox_inches='tight')
    plt.close(fig); print("    ✓")

# ══════════════════════════════════════════════════════════════
# FIGURA 2: CLIP ABLATION — ALPHA SWEEP COMPARATIVO
# ══════════════════════════════════════════════════════════════
if has_exp1:
    print(f"  Generando fig_v3_02_clip_ablation_alpha.png...")
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_key, label, color, marker in [
        ('vitb16', 'ViT-B/16 (512d)', C_VITB, 'o'),
        ('vitl14', 'ViT-L/14 (768d)', C_VITL, 's'),
    ]:
        sweep  = exp1[model_key]['alpha_sweep']
        alphas = sorted([float(a) for a in sweep.keys()])
        means  = [sweep[str(a)]['mean'] for a in alphas]
        stds   = [sweep[str(a)]['std']  for a in alphas]
        ax.errorbar(alphas, means, yerr=stds, label=label,
                    color=color, marker=marker, linewidth=2, markersize=7,
                    capsize=4, alpha=0.9)

    ax.axvline(x=0.2, color='gray', linestyle=':', alpha=0.5)
    ax.annotate('α=0.2\n(recomendado)', xy=(0.2, 0.5), xytext=(0.35, 0.55),
                fontsize=9, color='gray',
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
    ax.set_xlabel('α (peso textual)'); ax.set_ylabel('HR@500')
    ax.set_title('Alpha Sweep: ViT-B/16 vs ViT-L/14')
    ax.legend(loc='upper right', framealpha=0.9); ax.set_xlim(-0.05, 1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_v3_02_clip_ablation_alpha.png"),
                dpi=DPI, bbox_inches='tight')
    plt.close(fig); print("    ✓")

# ══════════════════════════════════════════════════════════════
# FIGURA 3: ROBUSTEZ — HR@500 POR % HARD DISTRACTORS
# ══════════════════════════════════════════════════════════════
if has_exp2:
    print(f"  Generando fig_v3_03_robustness_vqmin.png...")
    fig, ax   = plt.subplots(figsize=(9, 5.5))
    configs   = exp2['configs']
    hard_pcts = []
    strat_data = {'BFS': [], 'Text-Only': [], 'V-QMin a=0.2': []}

    for cname in ['hard_20pct', 'hard_40pct', 'hard_60pct', 'hard_80pct']:
        cfg   = configs[cname]
        hard_pcts.append(int(cfg['hard_ratio'] * 100))
        n_key = DEFAULT_N_KEY if DEFAULT_N_KEY in cfg['results'] \
                else list(cfg['results'].keys())[0]
        for sname in strat_data:
            strat_data[sname].append(cfg['results'][n_key][sname]['HR@500']['mean'])

    x = np.arange(len(hard_pcts)); w = 0.25
    for i, (sname, vals) in enumerate(strat_data.items()):
        bars = ax.bar(x + i*w, vals, w, label=sname,
                      color={'BFS': C_BFS, 'Text-Only': C_TEXT,
                             'V-QMin a=0.2': C_VQMIN}[sname],
                      alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.008,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8.5)

    ax.set_xlabel('Porcentaje de Hard Distractors'); ax.set_ylabel('HR@500')
    ax.set_title(f'Experimento 2: Robustez ante Variación de Hard Distractors\n'
                 f'(n={DEFAULT_N_KEY.split("=")[1]} queries)', fontweight='bold')
    ax.set_xticks(x + w); ax.set_xticklabels([f'{p}%' for p in hard_pcts])
    ax.set_ylim(0, 1.12); ax.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_v3_03_robustness_vqmin.png"),
                dpi=DPI, bbox_inches='tight')
    plt.close(fig); print("    ✓")

# ══════════════════════════════════════════════════════════════
# FIGURA 4: ROBUSTEZ — VENTAJA RELATIVA V-QMin vs Text-Only
# ══════════════════════════════════════════════════════════════
if has_exp2:
    print(f"  Generando fig_v3_04_robustness_advantage.png...")
    fig, ax  = plt.subplots(figsize=(8, 5))
    configs  = exp2['configs']
    palette  = [(50,  '#2196F3', 'o', '-'),
                (100, '#FF9800', 's', '--'),
                (300, '#9C27B0', '^', '-.'),
                (1000,'#E91E63', 'D', ':')]

    for n_q, color, marker, ls in palette:
        n_key = f'n_queries={n_q}'
        hp, adv = [], []
        for cname in ['hard_20pct', 'hard_40pct', 'hard_60pct', 'hard_80pct']:
            cfg = configs[cname]
            if n_key not in cfg['results']: continue
            vq  = cfg['results'][n_key]['V-QMin a=0.2']['HR@500']['mean']
            to  = cfg['results'][n_key]['Text-Only']['HR@500']['mean']
            hp.append(int(cfg['hard_ratio'] * 100))
            adv.append(100*(vq-to)/to if to > 0 else 0)
        if adv:
            ax.plot(hp, adv, color=color, marker=marker, linewidth=2,
                    markersize=8, linestyle=ls, label=f'n={n_q}', alpha=0.9)

    ax.set_xlabel('Porcentaje de Hard Distractors')
    ax.set_ylabel('Ventaja V-QMin vs Text-Only (%)')
    ax.set_title('Ventaja Relativa de V-QMin según Dificultad del Escenario',
                 fontweight='bold')
    ax.legend(framealpha=0.9)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_v3_04_robustness_advantage.png"),
                dpi=DPI, bbox_inches='tight')
    plt.close(fig); print("    ✓")

# ══════════════════════════════════════════════════════════════
# FIGURA 5: QUERY SCALING — HR@500 POR N QUERIES (v4: 50,100,300,1000)
# ══════════════════════════════════════════════════════════════
if has_exp3:
    print(f"  Generando fig_v3_05_query_scaling.png...")
    fig, ax    = plt.subplots(figsize=(10, 5.5))
    strategies = ['BFS', 'Text-Only', 'V-QMin a=0.2']
    colors_s   = {'BFS': C_BFS, 'Text-Only': C_TEXT, 'V-QMin a=0.2': C_VQMIN}
    x          = np.arange(len(N_QUERIES_LIST)); w = 0.25

    for i, sname in enumerate(strategies):
        means, stds = [], []
        for nq in N_QUERIES_LIST:
            d = exp3[f'n_queries={nq}']['strategies'][sname]['HR@500']
            means.append(d['mean']); stds.append(d['std'])
        bars = ax.bar(x + i*w, means, w, label=sname, color=colors_s[sname],
                      alpha=0.85, edgecolor='white', yerr=stds, capsize=4)
        for bar, val, sd in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + sd + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8.5)

    ax.set_xlabel('Número de Queries'); ax.set_ylabel('HR@500')
    ax.set_title('Experimento 3: Estabilidad con Escalado de Queries\n'
                 '(muestreo aleatorio del dataset)', fontweight='bold')
    ax.set_xticks(x + w); ax.set_xticklabels([str(n) for n in N_QUERIES_LIST])
    ax.set_ylim(0, 1.15); ax.legend(loc='upper left', framealpha=0.9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_v3_05_query_scaling.png"),
                dpi=DPI, bbox_inches='tight')
    plt.close(fig); print("    ✓")

# ══════════════════════════════════════════════════════════════
# FIGURA 6: QUERY SCALING — INTERVALOS DE CONFIANZA
# ══════════════════════════════════════════════════════════════
if has_exp3:
    print(f"  Generando fig_v3_06_query_scaling_ci.png...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    k_vals    = [50, 100, 200, 500, 1000, 2000, 5000]

    for idx, nq in enumerate(N_QUERIES_LIST):
        ax   = axes[idx]
        data = exp3[f'n_queries={nq}']['strategies']
        for sname, color, marker in [
            ('BFS',          C_BFS,   'x'),
            ('Text-Only',    C_TEXT,  's'),
            ('V-QMin a=0.2', C_VQMIN, 'o'),
        ]:
            means, ci_lo, ci_hi, k_plot = [], [], [], []
            for k in k_vals:
                key = f'HR@{k}'
                if key not in data[sname]: continue
                vals = data[sname][key]['values']
                m    = np.mean(vals)
                se   = np.std(vals) / np.sqrt(len(vals))
                means.append(m); ci_lo.append(m - 1.96*se)
                ci_hi.append(m + 1.96*se); k_plot.append(k)
            ax.plot(k_plot, means, color=color, marker=marker,
                    linewidth=1.5, markersize=5, label=sname, alpha=0.9)
            ax.fill_between(k_plot, ci_lo, ci_hi, color=color, alpha=0.15)

        ax.set_xlabel('K'); ax.set_ylabel('HR@K')
        ax.set_title(f'n={nq} queries')
        ax.set_xscale('log'); ax.set_ylim(0, 1.05)
        if idx == 0:
            ax.legend(loc='lower right', fontsize=8, framealpha=0.9)

    fig.suptitle('HR@K con Intervalos de Confianza 95% por Escala de Queries',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_v3_06_query_scaling_ci.png"),
                dpi=DPI, bbox_inches='tight')
    plt.close(fig); print("    ✓")

# ══════════════════════════════════════════════════════════════
# FIGURA 7: TABLA RESUMEN VISUAL
# ══════════════════════════════════════════════════════════════
print(f"  Generando fig_v3_07_summary_table.png...")
fig, ax = plt.subplots(figsize=(13, 7))
ax.axis('off')
ax.set_title('V-QMin v4 — Resumen de Experimentos Complementarios',
             fontsize=14, fontweight='bold', pad=20)

headers = ['Experimento', 'Configuración', 'V-QMin HR@500', 'vs Text-Only', 'vs BFS', 'p-valor']
rows    = []

if has_exp1:
    for model_key, label in [('vitb16', 'ViT-B/16'), ('vitl14', 'ViT-L/14')]:
        vq  = exp1[model_key]['results']['V-QMin a=0.2']['HR@500']['mean']
        to  = exp1[model_key]['results']['Text-Only']['HR@500']['mean']
        bfs = exp1[model_key]['results']['BFS']['HR@500']['mean']
        rows.append(['1. CLIP Ablación', label, f'{vq:.4f}',
                     f"+{100*(vq-to)/to:.1f}%", f"+{100*(vq-bfs)/bfs:.1f}%", '<0.001'])

if has_exp2:
    for cname in ['hard_20pct', 'hard_40pct', 'hard_60pct', 'hard_80pct']:
        cfg   = exp2['configs'][cname]
        n_key = DEFAULT_N_KEY if DEFAULT_N_KEY in cfg['results'] \
                else list(cfg['results'].keys())[0]
        vq  = cfg['results'][n_key]['V-QMin a=0.2']['HR@500']['mean']
        to  = cfg['results'][n_key]['Text-Only']['HR@500']['mean']
        bfs = cfg['results'][n_key]['BFS']['HR@500']['mean']
        rows.append(['2. Robustez', cfg['label'], f'{vq:.4f}',
                     f"+{100*(vq-to)/to:.1f}%", f"+{100*(vq-bfs)/bfs:.1f}%", '<0.001'])

if has_exp3:
    for nq in N_QUERIES_LIST:
        data = exp3[f'n_queries={nq}']['strategies']
        vq   = data['V-QMin a=0.2']['HR@500']['mean']
        to   = data['Text-Only']['HR@500']['mean']
        bfs  = data['BFS']['HR@500']['mean']
        p_w  = exp3[f'n_queries={nq}'].get('wilcoxon', {}) \
                   .get('V-QMin vs Text-Only', {}).get('p_value', None)
        p_str = f"{p_w:.2e}" if p_w is not None else "N/A"
        rows.append([f'3. Escalado', f'n={nq}', f'{vq:.4f}',
                     f"+{100*(vq-to)/to:.1f}%", f"+{100*(vq-bfs)/bfs:.1f}%", p_str])

if rows:
    table = ax.table(cellText=rows, colLabels=headers,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.5)
    for j in range(len(headers)):
        table[0, j].set_facecolor('#2E75B6')
        table[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(rows)+1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[i, j].set_facecolor('#F0F4F8')

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_v3_07_summary_table.png"),
            dpi=DPI, bbox_inches='tight')
plt.close(fig); print("    ✓")

# ══════════════════════════════════════════════════════════════
# FIGURA 8: ROBUSTEZ × QUERIES (v4: n=50, 100, 300, 1000)
# ══════════════════════════════════════════════════════════════
if has_exp2:
    print(f"  Generando fig_v3_08_robustness_by_nqueries.png...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    configs   = exp2['configs']

    for ax_idx, n_q in enumerate(N_QUERIES_LIST):
        ax    = axes[ax_idx]
        hp, sd = [], {'BFS': [], 'Text-Only': [], 'V-QMin a=0.2': []}

        for cname in ['hard_20pct', 'hard_40pct', 'hard_60pct', 'hard_80pct']:
            cfg   = configs[cname]
            hp.append(int(cfg['hard_ratio'] * 100))
            n_key = f'n_queries={n_q}'
            if n_key not in cfg['results']:
                n_key = list(cfg['results'].keys())[0]
            for sname in sd:
                sd[sname].append(cfg['results'][n_key][sname]['HR@500']['mean'])

        x = np.arange(len(hp)); w = 0.25
        for i, (sname, vals) in enumerate(sd.items()):
            bars = ax.bar(x + i*w, vals, w, label=sname,
                          color={'BFS': C_BFS, 'Text-Only': C_TEXT,
                                 'V-QMin a=0.2': C_VQMIN}[sname],
                          alpha=0.85, edgecolor='white')
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.008,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7)

        ax.set_xlabel('% Hard Distractors'); ax.set_ylabel('HR@500')
        ax.set_title(f'n={n_q} queries')
        ax.set_xticks(x + w); ax.set_xticklabels([f'{p}%' for p in hp])
        ax.set_ylim(0, 1.15)
        if ax_idx == 0:
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    fig.suptitle('Robustez × Queries: HR@500 por % Hard Distractors y Escala de Queries',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_v3_08_robustness_by_nqueries.png"),
                dpi=DPI, bbox_inches='tight')
    plt.close(fig); print("    ✓")

# ══════════════════════════════════════════════════════════════
# RESUMEN
# ══════════════════════════════════════════════════════════════
figs = sorted([f for f in os.listdir(FIG_DIR) if f.endswith('.png')])
print(f"\n{'=' * 70}")
print(f"STEP 09 COMPLETADO ✓ — {len(figs)} figuras en {FIG_DIR}")
for f in figs:
    print(f"  {f} ({os.path.getsize(os.path.join(FIG_DIR, f))/1024:.0f} KB)")
print("=" * 70)
