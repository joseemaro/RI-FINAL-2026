# V-QMin — Paquete de Replicación Completo

## Descripción

Este repo permite replicar los experimentos del paper V-QMin
desde los parquets crudos de VisualWebInstruct-verified.

**Paper:** *V-QMin: Un Web Crawler Multimodal con Señales Visuales para
Recuperación de Información Académica*
**Autor:** Jose Emanuel Rodriguez — Universidad Nacional de Luján, Curso 11090

---

## Requisitos

### Hardware
- GPU NVIDIA con ≥8 GB VRAM (testeado en RTX 5060 Ti 16 GB)
- 16 GB RAM
- ≥5 GB disco para outputs

### Software
```
Python >= 3.10
PyTorch >= 2.0 (con CUDA)
open-clip-torch
pyarrow
pandas
numpy
scipy
matplotlib
seaborn
Pillow
```

Instalar:
```bash
pip install open-clip-torch pyarrow pandas numpy scipy matplotlib seaborn Pillow
```

### Dataset
**TIGER-Lab/VisualWebInstruct-verified** en parquets locales.

Se puede descargar clonando el siguiente repositorio: https://huggingface.co/datasets/TIGER-Lab/VisualWebInstruct-verified

Estructura actual:
```
I:\RIFINALV3\
├── batch_0001_of_0022\
│   └── *.parquet
├── batch_0002_of_0022\
│   └── *.parquet
├── ...
└── batch_0022_of_0022\
    └── *.parquet
```

---

## Orden de Ejecución

```
python step_01_explore_dataset.py        # ~2 min   — Validar parquets
python step_02_gpu_clip_validation.py    # ~1 min   — Validar GPU + CLIP
python step_03_extract_targets.py        # ~5 min   — 10K targets + embeddings ViT-B/16
python step_04_build_graph.py            # ~2 min   — Grafo sintético 50K nodos
python step_05_evaluate.py               # ~15 min  — Crawlers + métricas + Wilcoxon
python step_06_figures_main.py           # ~1 min   — 8 figuras del experimento principal
python step_07_vitl14_embeddings.py      # ~10 min  — Embeddings ViT-L/14 (ablación)
python step_08_complementary_exp.py      # ~30 min  — 3 experimentos complementarios
python step_09_figures_complementary.py  # ~1 min   — 8 figuras complementarias
```

**Tiempo total estimado: ~65 minutos** (RTX 5060 Ti).

Cada script verifica que sus dependencias existan antes de ejecutar.
Si un paso falla, corregir y re-ejecutar desde ese paso.

---

## Dependencias entre Scripts

```
step_01 → step_02 → step_03 → step_04 → step_05 → step_06
                         │                   │
                         └──→ step_07 ──→ step_08 → step_09
```

- Steps 01-06: Pipeline principal (evaluación base)
- Step 07: Requiere step_03 (re-usa los mismos registros)
- Step 08: Requiere steps 03, 04, 05, 07
- Step 09: Requiere step 08

---

## Outputs Generados

Directorio: `I:\RIFINALV3\vqmin_outputs\`

### Datos intermedios
| Archivo | Descripción |
|---------|-------------|
| `fase01_exploration.json` | Estadísticas del dataset |
| `fase02_gpu_clip.json` | Validación GPU + CLIP |
| `targets_10k.pkl` | 10K targets con embeddings ViT-B/16 |
| `img_embeddings_10k.npy` | Embeddings imagen (10000, 512) |
| `txt_embeddings_10k.npy` | Embeddings texto (10000, 512) |
| `synthetic_graph_50k.pkl` | Grafo completo |
| `evaluation_results.json` | Métricas principales |
| `alpha_sweep.json` | Barrido de α |
| `vitl14_img_embeddings_10k.npy` | Embeddings ViT-L/14 imagen (10000, 768) |
| `vitl14_txt_embeddings_10k.npy` | Embeddings ViT-L/14 texto (10000, 768) |
| `exp1_clip_ablation.json` | Resultados ablación CLIP |
| `exp2_distractor_robustness.json` | Resultados robustez |
| `exp3_query_scaling.json` | Resultados query scaling |

### Figuras (300 DPI, PNG)
| Figura | Contenido |
|--------|-----------|
| `figures_main/fig1_hr_at_k.png` | HR@K comparativo |
| `figures_main/fig2_ndcg_at_k.png` | nDCG@K comparativo |
| `figures_main/fig3_alpha_sweep.png` | Barrido de α |
| `figures_main/fig4_bars_hr500.png` | Barras HR@500 |
| `figures_main/fig5_heatmap_queries.png` | Heatmap por query |
| `figures_main/fig6_relative_improvement.png` | Mejora relativa |
| `figures_main/fig7_similarity_distribution.png` | Distribución similitudes |
| `figures_main/fig8_results_table.png` | Tabla resumen |
| `figures_v3/fig_v3_01_clip_ablation_bars.png` | Ablación barras |
| `figures_v3/fig_v3_02_clip_ablation_alpha.png` | Ablación alpha sweep |
| `figures_v3/fig_v3_03_robustness_vqmin.png` | Robustez barras |
| `figures_v3/fig_v3_04_robustness_advantage.png` | Ventaja relativa |
| `figures_v3/fig_v3_05_query_scaling.png` | Query scaling barras |
| `figures_v3/fig_v3_06_query_scaling_ci.png` | Query scaling CI |
| `figures_v3/fig_v3_07_summary_table.png` | Tabla resumen v3 |
| `figures_v3/fig_v3_08_robustness_by_nqueries.png` | Robustez × queries |

---

## Configuración de Paths

Todos los scripts usan estas variables al inicio:

```python
DATASET_DIR = r"I:\RIFINALV3"
OUTPUT_DIR  = os.path.join(DATASET_DIR, "vqmin_outputs")
```

Para cambiar la ubicación del dataset, editar `DATASET_DIR` en cada script.

---

## Parámetros del Experimento

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| N_TARGETS | 10,000 | Targets reales del dataset |
| DISTRACTOR_RATIO | 4:1 | 40,000 distractores |
| HARD_RATIO | 0.40 | 40% hard, 60% easy |
| HARD_NOISE_T1 | σ=0.05 | Tier 1 (sim ~0.65-0.75) |
| HARD_NOISE_T2 | σ=0.10 | Tier 2 (sim ~0.40-0.55) |
| α (V-QMin) | 0.2 | Peso textual óptimo |
| SEED | 42 | Reproducibilidad |
| CLIP modelo | ViT-B-16 | Encoder principal |
| CLIP ablación | ViT-L-14 | Encoder alternativo |
| Queries base | 10 STEM | Evaluación principal |
| Queries total | 50 STEM | Exp. complementarios |

---

## Reproducibilidad

- Todos los scripts usan `SEED = 42`
- `np.random.seed(42)`, `random.seed(42)`, `torch.manual_seed(42)`
- El sampleo estratificado garantiza la misma selección de targets
- La construcción del grafo usa `np.random.RandomState(42)`

---
