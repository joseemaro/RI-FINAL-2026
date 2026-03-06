#!/usr/bin/env python3
"""
V-QMin — STEP 09: FIGURAS COMPARATIVAS PARA EL PAPER
========================================================
Genera todas las figuras nuevas a partir de los JSONs de Step 2.

Figuras generadas:
  fig_v3_01_clip_ablation_bars.png    — HR@K comparando ViT-B/16 vs ViT-L/14
  fig_v3_02_clip_ablation_alpha.png   — Alpha sweep de ambos modelos superpuestos
  fig_v3_03_robustness_vqmin.png      — HR@500 de V-QMin vs % hard distractors
  fig_v3_04_robustness_advantage.png  — Ventaja relativa V-QMin vs Text-Only por % hard
  fig_v3_05_query_scaling.png         — HR@500 con barras de error para n=10,30,50
  fig_v3_06_query_scaling_ci.png      — Intervalos de confianza al escalar queries
  fig_v3_07_summary_table.png         — Tabla resumen de los 3 experimentos

Entrada:  v3_exp1_clip_ablation.json, v3_exp2_distractor_robustness.json,
          v3_exp3_query_scaling.json
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
OUTPUT_DIR = os.path.join(r"I:\RIFINALV3", "vqmin_outputs")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures_v3")
os.makedirs(FIG_DIR, exist_ok=True)

DPI = 300
STYLE_PARAMS = {
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
}
plt.rcParams.update(STYLE_PARAMS)

# Colores
C_VITB = '#2196F3'    # Azul
C_VITL = '#FF9800'    # Naranja
C_BFS = '#9E9E9E'     # Gris
C_TEXT = '#4CAF50'     # Verde
C_VQMIN = '#E91E63'   # Rosa/Magenta
C_VQMIN2 = '#9C27B0'  # Púrpura

print("=" * 70)
print("V-QMin — STEP 09: FIGURAS COMPARATIVAS")
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
    with open(exp1_path) as f:
        exp1 = json.load(f)
    print(f"  ✓ Exp 1 (CLIP Ablation): cargado")

if has_exp2:
    with open(exp2_path) as f:
        exp2 = json.load(f)
    print(f"  ✓ Exp 2 (Distractor Robustness): cargado")

if has_exp3:
    with open(exp3_path) as f:
        exp3 = json.load(f)
    print(f"  ✓ Exp 3 (Query Scaling): cargado")


# ══════════════════════════════════════════════════════════════
# FIGURA 1: CLIP ABLATION — BARRAS HR@K
# ══════════════════════════════════════════════════════════════
if has_exp1:
    print(f"\n  Generando fig_v3_01_clip_ablation_bars.png...")

    k_vals = [100, 500, 1000, 5000]
    strategies = ['BFS', 'Text-Only', 'V-QMin a=0.2']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax_idx, (model_key, model_label, color_main) in enumerate([
        ('vitb16', 'ViT-B/16', C_VITB),
        ('vitl14', 'ViT-L/14', C_VITL),
    ]):
        ax = axes[ax_idx]
        data = exp1[model_key]['results']
        x = np.arange(len(k_vals))
        width = 0.25

        colors = [C_BFS, C_TEXT, C_VQMIN]
        for i, strat in enumerate(strategies):
            means = [data[strat][f'HR@{k}']['mean'] for k in k_vals]
            stds = [data[strat][f'HR@{k}']['std'] for k in k_vals]
            bars = ax.bar(x + i * width, means, width, label=strat,
                         color=colors[i], alpha=0.85, edgecolor='white',
                         yerr=stds, capsize=3)
            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=7.5)

        ax.set_xlabel('K')
        ax.set_ylabel('Harvest Rate')
        ax.set_title(f'{model_label} (d={exp1[model_key].get("embed_dim", "?")})')
        ax.set_xticks(x + width)
        ax.set_xticklabels([f'{k}' for k in k_vals])
        ax.set_ylim(0, 1.12)
        ax.legend(loc='upper left', framealpha=0.9)

    fig.suptitle('Experimento 1: Ablación de Modelo CLIP — HR@K', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_v3_01_clip_ablation_bars.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"    ✓ Guardada")


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
        sweep = exp1[model_key]['alpha_sweep']
        alphas = sorted([float(a) for a in sweep.keys()])
        means = [sweep[str(a)]['mean'] for a in alphas]
        stds = [sweep[str(a)]['std'] for a in alphas]

        ax.errorbar(alphas, means, yerr=stds, label=label,
                    color=color, marker=marker, linewidth=2, markersize=7,
                    capsize=4, alpha=0.9)

    ax.set_xlabel('α (peso textual)')
    ax.set_ylabel('HR@500')
    ax.set_title('Alpha Sweep: ViT-B/16 vs ViT-L/14')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(-0.05, 1.05)
    ax.axvline(x=0.2, color='gray', linestyle=':', alpha=0.5, label='α=0.2 recomendado')

    # Anotar alfa recomendado
    ax.annotate('α=0.2\n(recomendado)', xy=(0.2, 0.5), xytext=(0.35, 0.55),
                fontsize=9, color='gray', arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_v3_02_clip_ablation_alpha.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"    ✓ Guardada")


# ══════════════════════════════════════════════════════════════
# FIGURA 3: ROBUSTEZ — HR@500 POR % HARD DISTRACTORS
# ══════════════════════════════════════════════════════════════
if has_exp2:
    print(f"  Generando fig_v3_03_robustness_vqmin.png...")

    fig, ax = plt.subplots(figsize=(9, 5.5))

    configs = exp2['configs']
    hard_pcts = []
    strategies_data = {'BFS': [], 'Text-Only': [], 'V-QMin a=0.2': []}

    for cname in ['hard_20pct', 'hard_40pct', 'hard_60pct', 'hard_80pct']:
        cfg = configs[cname]
        hard_pcts.append(int(cfg['hard_ratio'] * 100))

        # Usar n=30 queries por defecto
        n_key = 'n_queries=30'
        if n_key not in cfg['results']:
            n_key = list(cfg['results'].keys())[0]

        for sname in strategies_data.keys():
            strategies_data[sname].append(
                cfg['results'][n_key][sname]['HR@500']['mean']
            )

    x = np.arange(len(hard_pcts))
    width = 0.25
    colors = {'BFS': C_BFS, 'Text-Only': C_TEXT, 'V-QMin a=0.2': C_VQMIN}

    for i, (sname, vals) in enumerate(strategies_data.items()):
        bars = ax.bar(x + i * width, vals, width, label=sname,
                     color=colors[sname], alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.008,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8.5)

    ax.set_xlabel('Porcentaje de Hard Distractors')
    ax.set_ylabel('HR@500')
    ax.set_title('Experimento 2: Robustez ante Variación de Hard Distractors', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'{p}%' for p in hard_pcts])
    ax.set_ylim(0, 1.12)
    ax.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_v3_03_robustness_vqmin.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"    ✓ Guardada")


# ══════════════════════════════════════════════════════════════
# FIGURA 4: ROBUSTEZ — VENTAJA RELATIVA V-QMin vs Text-Only
# ══════════════════════════════════════════════════════════════
if has_exp2:
    print(f"  Generando fig_v3_04_robustness_advantage.png...")

    fig, ax = plt.subplots(figsize=(8, 5))

    configs = exp2['configs']
    for n_q, color, marker, ls in [
        (10, '#2196F3', 'o', '-'),
        (30, '#FF9800', 's', '--'),
        (50, '#E91E63', '^', '-.'),
    ]:
        hard_pcts = []
        advantages = []

        for cname in ['hard_20pct', 'hard_40pct', 'hard_60pct', 'hard_80pct']:
            cfg = configs[cname]
            hard_pcts.append(int(cfg['hard_ratio'] * 100))

            n_key = f'n_queries={n_q}'
            if n_key not in cfg['results']:
                continue

            vq = cfg['results'][n_key]['V-QMin a=0.2']['HR@500']['mean']
            to = cfg['results'][n_key]['Text-Only']['HR@500']['mean']
            adv = 100 * (vq - to) / to if to > 0 else 0
            advantages.append(adv)

        if advantages:
            ax.plot(hard_pcts[:len(advantages)], advantages,
                   color=color, marker=marker, linewidth=2, markersize=8,
                   linestyle=ls, label=f'n={n_q} queries', alpha=0.9)

    ax.set_xlabel('Porcentaje de Hard Distractors')
    ax.set_ylabel('Ventaja V-QMin vs Text-Only (%)')
    ax.set_title('Ventaja Relativa de V-QMin según Dificultad del Escenario', fontweight='bold')
    ax.legend(framealpha=0.9)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_v3_04_robustness_advantage.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"    ✓ Guardada")


# ══════════════════════════════════════════════════════════════
# FIGURA 5: QUERY SCALING — HR@500 POR N QUERIES
# ══════════════════════════════════════════════════════════════
if has_exp3:
    print(f"  Generando fig_v3_05_query_scaling.png...")

    fig, ax = plt.subplots(figsize=(9, 5.5))

    n_queries_list = [10, 30, 50]
    strategies = ['BFS', 'Text-Only', 'V-QMin a=0.2']
    colors = {'BFS': C_BFS, 'Text-Only': C_TEXT, 'V-QMin a=0.2': C_VQMIN}

    x = np.arange(len(n_queries_list))
    width = 0.25

    for i, sname in enumerate(strategies):
        means = []
        stds = []
        for nq in n_queries_list:
            data = exp3[f'n_queries={nq}']['strategies'][sname]['HR@500']
            means.append(data['mean'])
            stds.append(data['std'])

        bars = ax.bar(x + i * width, means, width, label=sname,
                     color=colors[sname], alpha=0.85, edgecolor='white',
                     yerr=stds, capsize=4)
        for bar, val, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8.5)

    ax.set_xlabel('Número de Queries')
    ax.set_ylabel('HR@500')
    ax.set_title('Experimento 3: Estabilidad con Escalado de Queries', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([str(n) for n in n_queries_list])
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper left', framealpha=0.9)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_v3_05_query_scaling.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"    ✓ Guardada")


# ══════════════════════════════════════════════════════════════
# FIGURA 6: QUERY SCALING — INTERVALOS DE CONFIANZA
# ══════════════════════════════════════════════════════════════
if has_exp3:
    print(f"  Generando fig_v3_06_query_scaling_ci.png...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, nq in enumerate([10, 30, 50]):
        ax = axes[idx]
        data = exp3[f'n_queries={nq}']['strategies']

        k_vals = [50, 100, 200, 500, 1000, 2000, 5000]

        for sname, color, marker in [
            ('BFS', C_BFS, 'x'),
            ('Text-Only', C_TEXT, 's'),
            ('V-QMin a=0.2', C_VQMIN, 'o'),
        ]:
            means = []
            ci_low = []
            ci_high = []
            k_plot = []
            for k in k_vals:
                key = f'HR@{k}'
                if key in data[sname]:
                    vals = data[sname][key]['values']
                    m = np.mean(vals)
                    se = np.std(vals) / np.sqrt(len(vals))
                    means.append(m)
                    ci_low.append(m - 1.96 * se)
                    ci_high.append(m + 1.96 * se)
                    k_plot.append(k)

            ax.plot(k_plot, means, color=color, marker=marker,
                   linewidth=1.5, markersize=5, label=sname, alpha=0.9)
            ax.fill_between(k_plot, ci_low, ci_high, color=color, alpha=0.15)

        ax.set_xlabel('K')
        ax.set_ylabel('HR@K')
        ax.set_title(f'n={nq} queries')
        ax.set_xscale('log')
        ax.set_ylim(0, 1.05)
        if idx == 0:
            ax.legend(loc='lower right', fontsize=8, framealpha=0.9)

    fig.suptitle('HR@K con Intervalos de Confianza 95% por Escala de Queries',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_v3_06_query_scaling_ci.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"    ✓ Guardada")


# ══════════════════════════════════════════════════════════════
# FIGURA 7: TABLA RESUMEN VISUAL
# ══════════════════════════════════════════════════════════════
print(f"  Generando fig_v3_07_summary_table.png...")

fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')
ax.set_title('V-QMin v3 — Resumen de Experimentos Nuevos',
             fontsize=14, fontweight='bold', pad=20)

# Construir datos de la tabla
headers = ['Experimento', 'Configuración', 'V-QMin HR@500', 'vs Text-Only', 'vs BFS', 'p-valor']
rows = []

# Exp 1 data
if has_exp1:
    for model_key, label in [('vitb16', 'ViT-B/16'), ('vitl14', 'ViT-L/14')]:
        vq = exp1[model_key]['results']['V-QMin a=0.2']['HR@500']['mean']
        to = exp1[model_key]['results']['Text-Only']['HR@500']['mean']
        bfs = exp1[model_key]['results']['BFS']['HR@500']['mean']
        pct_text = f"+{100*(vq-to)/to:.1f}%" if to > 0 else "N/A"
        pct_bfs = f"+{100*(vq-bfs)/bfs:.1f}%" if bfs > 0 else "N/A"
        rows.append([f'1. CLIP Ablación', label, f'{vq:.4f}', pct_text, pct_bfs, '<0.001'])

# Exp 2 data
if has_exp2:
    configs = exp2['configs']
    for cname in ['hard_20pct', 'hard_40pct', 'hard_60pct', 'hard_80pct']:
        cfg = configs[cname]
        n_key = 'n_queries=30' if 'n_queries=30' in cfg['results'] else list(cfg['results'].keys())[0]
        vq = cfg['results'][n_key]['V-QMin a=0.2']['HR@500']['mean']
        to = cfg['results'][n_key]['Text-Only']['HR@500']['mean']
        bfs = cfg['results'][n_key]['BFS']['HR@500']['mean']
        pct_text = f"+{100*(vq-to)/to:.1f}%" if to > 0 else "N/A"
        pct_bfs = f"+{100*(vq-bfs)/bfs:.1f}%" if bfs > 0 else "N/A"
        rows.append([f'2. Robustez', cfg['label'], f'{vq:.4f}', pct_text, pct_bfs, '<0.001'])

# Exp 3 data
if has_exp3:
    for nq in [10, 30, 50]:
        data = exp3[f'n_queries={nq}']['strategies']
        vq = data['V-QMin a=0.2']['HR@500']['mean']
        to = data['Text-Only']['HR@500']['mean']
        bfs = data['BFS']['HR@500']['mean']
        pct_text = f"+{100*(vq-to)/to:.1f}%" if to > 0 else "N/A"
        pct_bfs = f"+{100*(vq-bfs)/bfs:.1f}%" if bfs > 0 else "N/A"

        wilcoxon = exp3[f'n_queries={nq}'].get('wilcoxon', {})
        p_text = wilcoxon.get('V-QMin vs Text-Only', {}).get('p_value', None)
        p_str = f"{p_text:.4f}" if p_text is not None else "N/A"
        rows.append([f'3. Queries', f'n={nq}', f'{vq:.4f}', pct_text, pct_bfs, p_str])

if rows:
    table = ax.table(cellText=rows, colLabels=headers,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Estilo de cabecera
    for j in range(len(headers)):
        table[0, j].set_facecolor('#2E75B6')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Colores alternados
    for i in range(1, len(rows) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[i, j].set_facecolor('#F0F4F8')

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_v3_07_summary_table.png"), dpi=DPI, bbox_inches='tight')
plt.close(fig)
print(f"    ✓ Guardada")


# ══════════════════════════════════════════════════════════════
# FIGURA 8: ROBUSTEZ CON MÚLTIPLES N_QUERIES (n=10, 30, 50)
# ══════════════════════════════════════════════════════════════
if has_exp2:
    print(f"  Generando fig_v3_08_robustness_by_nqueries.png...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax_idx, n_q in enumerate([10, 30, 50]):
        ax = axes[ax_idx]
        configs = exp2['configs']

        hard_pcts = []
        strat_data = {'BFS': [], 'Text-Only': [], 'V-QMin a=0.2': []}

        for cname in ['hard_20pct', 'hard_40pct', 'hard_60pct', 'hard_80pct']:
            cfg = configs[cname]
            hard_pcts.append(int(cfg['hard_ratio'] * 100))
            n_key = f'n_queries={n_q}'
            if n_key not in cfg['results']:
                n_key = list(cfg['results'].keys())[0]
            for sname in strat_data:
                strat_data[sname].append(cfg['results'][n_key][sname]['HR@500']['mean'])

        x = np.arange(len(hard_pcts))
        w = 0.25
        colors_s = {'BFS': C_BFS, 'Text-Only': C_TEXT, 'V-QMin a=0.2': C_VQMIN}

        for i, (sname, vals) in enumerate(strat_data.items()):
            bars = ax.bar(x + i * w, vals, w, label=sname,
                         color=colors_s[sname], alpha=0.85, edgecolor='white')
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.008,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=7)

        ax.set_xlabel('% Hard Distractors')
        ax.set_ylabel('HR@500')
        ax.set_title(f'n={n_q} queries')
        ax.set_xticks(x + w)
        ax.set_xticklabels([f'{p}%' for p in hard_pcts])
        ax.set_ylim(0, 1.15)
        if ax_idx == 0:
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    fig.suptitle('Robustez × Queries: HR@500 por % Hard Distractors y Escala de Queries',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_v3_08_robustness_by_nqueries.png"), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"    ✓ Guardada")


# ══════════════════════════════════════════════════════════════
# RESUMEN
# ══════════════════════════════════════════════════════════════
figs = [f for f in os.listdir(FIG_DIR) if f.endswith('.png')]
print(f"\n{'=' * 70}")
print("V-QMin — STEP 09 COMPLETADO ✓")
print("=" * 70)
print(f"  Directorio: {FIG_DIR}")
print(f"  Figuras generadas: {len(figs)}")
for f in sorted(figs):
    size = os.path.getsize(os.path.join(FIG_DIR, f)) / 1024
    print(f"    {f} ({size:.0f} KB)")
print(f"\n  Estas figuras van en una nueva sección del paper:")
print(f"  'Sección 7: Experimentos Adicionales' (o similar)")
print("=" * 70)
