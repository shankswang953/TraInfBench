#!/usr/bin/env python3
"""All-seven-time paired-metacell atlas with a frozen RNA-to-ATAC map."""
from __future__ import annotations

# Repository layout adapter; scientific calculations below are retained.
import sys as _sys
from pathlib import Path as _RepoPath
_REPO = _RepoPath(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO / "common"))
from benchmark_runtime import configure as _configure
_configure(_REPO)


import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

# Match the established shared scientific environment without system changes.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-cache')
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/trainfbench-hc-umap-numba')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/trainfbench-hc-umap-xdg')

import anndata as ad
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path('external/COATI/humanCerebral/Data')
DATA = DATA_ROOT / 'selected_4_7_9_11_12_18_21'
T_DIR = DATA_ROOT / 'TrainT_7time_D4_D21_no_D16_lsi12'
CHECKPOINT = T_DIR / 'T_FiLM.pt'
EXPECTED_T_SHA = 'bd32676dfd2732c2eb2484a54972cac09c576d3b413346bcce09f3c44ddacb0a'
MODEL_SOURCE = DATA_ROOT / 'TrainT/map_models.py'
METADATA = ROOT / 'data/human_cerebral_7time_d4_d21_no_d16_rna_cytobridge_balanced.h5ad'
OBS_COORDS = ROOT / 'results/human_cerebral_original_metacell_umaps/umap_coordinates.csv.gz'
UMAP_MODEL = ROOT / 'results/human_cerebral_original_atac_frozen_lsi12_umap/metacell_model_lsi12_umap_model.joblib'
FONT = Path(__import__('matplotlib.font_manager', fromlist=['findfont']).findfont('Arial'))
DEFAULT_OUTPUT = ROOT / 'results/human_cerebral_t_mapping_reference_v1'
DAYS = (4, 7, 9, 11, 12, 18, 21)
LABELS = ('early', 'nt', 'telencephalon', 'other')
DISPLAY = {'early': 'Early', 'nt': 'Non-telencephalon', 'telencephalon': 'Telencephalon', 'other': 'Other'}
COLORS = {'early': '#E6AB02', 'nt': '#1B9E77', 'telencephalon': '#7570B3', 'other': '#999999'}
EXPECTED_COUNTS = {'early': 7308, 'nt': 2739, 'telencephalon': 10581, 'other': 35}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_blocks(path, ids=False):
    with np.load(path, allow_pickle=ids) as archive:
        return [np.asarray(archive[f'age_{day}'], dtype=str if ids else np.float32) for day in DAYS]


def limits(values):
    lower = values.min(axis=0).astype(float)
    upper = values.max(axis=0).astype(float)
    padding = np.maximum(upper - lower, 1e-6) * .04
    return np.stack([lower - padding, upper + padding])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--batch-size', type=int, default=2048)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise SystemExit(f'Refusing existing output: {out}')
    inputs = [CHECKPOINT, MODEL_SOURCE, METADATA, OBS_COORDS, UMAP_MODEL,
              DATA / 'rna_pca30_normalized_by_time.npz', DATA / 'atac_lsi12_normalized_by_time.npz',
              DATA / 'rna_pca30_by_time.npz', DATA / 'atac_lsi12_by_time.npz',
              DATA / 'paired_metacell_ids_by_time.npz', DATA / 'rna_normalization.json',
              DATA / 'atac_lsi12_normalization.json', T_DIR / 'split_indices.npz',
              T_DIR / 'split_metadata.json', T_DIR / 'training_summary.json', FONT,
              ROOT / 'comparison/human_cerebral/plot_human_cerebral_original_metacell_umaps.py',
              ROOT / 'comparison/human_cerebral/prepare_human_cerebral_original_atac_lsi_umap.py',
              ROOT / 'comparison/palate/plot_palate_rna_atac_celltype_reference.py',
              DATA_ROOT / 'TrainT/train_models.py', Path(__file__).resolve()]
    before = [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)} for path in inputs]
    assert sha(CHECKPOINT) == EXPECTED_T_SHA
    x_blocks = load_blocks(DATA / 'rna_pca30_normalized_by_time.npz')
    y_blocks = load_blocks(DATA / 'atac_lsi12_normalized_by_time.npz')
    id_blocks = load_blocks(DATA / 'paired_metacell_ids_by_time.npz', ids=True)
    for day, x, y, ids in zip(DAYS, x_blocks, y_blocks, id_blocks):
        assert len(x) == len(y) == len(ids), f'Pair count at D{day}'
    x, y, ids = np.vstack(x_blocks), np.vstack(y_blocks), np.concatenate(id_blocks)
    days = np.concatenate([np.full(len(block), day, dtype=np.int16) for day, block in zip(DAYS, x_blocks)])
    times = ((days.astype(np.float64) - 4) / 10).astype(np.float32)
    assert x.shape == (20663, 30) and y.shape == (20663, 12)
    assert len(np.unique(ids)) == 20663 and np.isfinite(x).all() and np.isfinite(y).all()
    metadata = ad.read_h5ad(METADATA)
    assert np.array_equal(metadata.obs_names.to_numpy(str), ids)
    assert np.array_equal(metadata.obs.paired_metacell_id.to_numpy(str), ids)
    assert np.array_equal(metadata.obs.age_day.to_numpy(int), days)
    assert np.array_equal(metadata.obs.time_continuous.to_numpy(np.float32), times)
    assert np.array_equal(np.asarray(metadata.obsm['X_latent'], np.float32), x)
    labels = metadata.obs.lineage_coarse.to_numpy(str)
    lines = metadata.obs.line.to_numpy(str)
    counts = {label: int(np.sum(labels == label)) for label in LABELS}
    assert counts == EXPECTED_COUNTS and set(labels) == set(LABELS)
    normalization = {mod: json.loads((DATA / name).read_text()) for mod, name in [('RNA', 'rna_normalization.json'), ('ATAC', 'atac_lsi12_normalization.json')]}
    norm_audit = {}
    for mod, raw_file, normalized in [('RNA', 'rna_pca30_by_time.npz', x), ('ATAC', 'atac_lsi12_by_time.npz', y)]:
        raw = np.vstack(load_blocks(DATA / raw_file))
        error = float(np.max(np.abs(raw / normalization[mod]['scale'] - normalized)))
        assert error <= 1e-6
        norm_audit[mod + '_raw_divided_by_scale_vs_normalized_max_abs'] = error
    coords = pd.read_csv(OBS_COORDS)
    embeddings = {}
    for rep, space, mod in [('Model RNA', 'corrected_normalized_PCA30', 'RNA'), ('Model ATAC', 'corrected_normalized_LSI12', 'ATAC')]:
        frame = coords.loc[coords.representation == rep]
        assert len(frame) == 20663 and frame.id.is_unique
        assert set(frame.space) == {space}
        assert np.array_equal(frame.id.to_numpy(str), ids)
        assert np.array_equal(frame.day.to_numpy(int), days)
        assert np.array_equal(frame.line.str.lower().to_numpy(str), np.char.lower(lines))
        embeddings[mod] = frame[['UMAP1', 'UMAP2']].to_numpy(np.float32)
    reducer = joblib.load(UMAP_MODEL)
    raw_error = float(np.max(np.abs(np.asarray(reducer._raw_data, np.float32) - y)))
    embedding_error = float(np.max(np.abs(np.asarray(reducer.embedding_, np.float32) - embeddings['ATAC'])))
    assert raw_error == 0 and embedding_error == 0
    assert reducer.n_neighbors == 30 and reducer.min_dist == .25
    assert reducer.random_state == 42 and reducer.transform_seed == 42 and reducer.metric == 'euclidean'
    print('Identity, normalized space and frozen observed UMAP audits passed.', flush=True)
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    spec = importlib.util.spec_from_file_location('hc_frozen_film_for_reference', MODEL_SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    model = module.FiLMMLP(**checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval().requires_grad_(False)
    torch.set_num_threads(1)
    assert checkpoint['config']['t_min'] == 0 and checkpoint['config']['t_max'] == 1.7
    for mod, key in [('RNA', 'rna_normalization_scale'), ('ATAC', 'atac_normalization_scale')]:
        assert checkpoint['meta'][key] == normalization[mod]['scale']
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), args.batch_size):
            stop = min(start + args.batch_size, len(x))
            chunks.append(model(torch.from_numpy(x[start:stop]), torch.from_numpy(times[start:stop])).numpy())
    prediction = np.concatenate(chunks).astype(np.float32, copy=False)
    assert prediction.shape == y.shape and np.isfinite(prediction).all()
    # The model already restores its internal target standardization. Its output
    # remains in the external normalized LSI12 model space; do not rescale again.
    split_labels = np.full(len(x), '', dtype='<U5')
    test_sanity = []
    with np.load(T_DIR / 'split_indices.npz', allow_pickle=False) as archive:
        offset = 0
        for day, xb in zip(DAYS, x_blocks):
            pieces = [archive[f'age_{day}_{split}'] for split in ('train', 'val', 'test')]
            assert np.array_equal(np.sort(np.concatenate(pieces)), np.arange(len(xb)))
            for split, indices in zip(('train', 'val', 'test'), pieces):
                split_labels[offset + indices] = split
            test_index = offset + pieces[2]
            err = ((torch.from_numpy(prediction[test_index] - y[test_index]) / model.y_scale) ** 2).mean(dim=1)
            mse = float(err.sum()) / len(test_index)
            expected = float(checkpoint['meta']['test_per_time_standardized_mse'][f'age_{day}'])
            test_sanity.append({'day': day, 'n_test': len(test_index), 'paired_target_standardized_mse': mse, 'checkpoint_record': expected, 'absolute_difference': abs(mse - expected)})
            offset += len(xb)
    equal_time_mse = float(np.mean([row['paired_target_standardized_mse'] for row in test_sanity]))
    expected_mse = float(checkpoint['meta']['test_equal_time_standardized_mse'])
    assert abs(equal_time_mse - expected_mse) < 2e-6
    assert max(row['absolute_difference'] for row in test_sanity) < 2e-6
    print(f'Frozen T test-split sanity MSE: {equal_time_mse:.12f}; recorded {expected_mse:.12f}.', flush=True)
    print('Transforming predictions through the frozen observed-ATAC UMAP; no refit.', flush=True)
    predicted_umap = reducer.transform(prediction).astype(np.float32)
    assert predicted_umap.shape == (20663, 2) and np.isfinite(predicted_umap).all()
    rna_limits = limits(embeddings['RNA'])
    atac_limits = limits(np.vstack([embeddings['ATAC'], predicted_umap]))
    fontManager.addfont(str(FONT))
    font = FontProperties(fname=str(FONT), size=10)
    assert font.get_name() == 'Arial'
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'axes.titlesize': 10, 'legend.fontsize': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.unicode_minus': False, 'savefig.facecolor': 'white'})
    fig = plt.figure(figsize=(200 / 25.4, 60 / 25.4), facecolor='white')
    grid = fig.add_gridspec(1, 4, width_ratios=(1, 1, 1, 1.0), wspace=.07)
    draw_order = sorted(LABELS, key=lambda label: counts[label], reverse=True)
    plot_checks = []
    for i, (title, xy, lim) in enumerate([('RNA space', embeddings['RNA'], rna_limits), ('ATAC space', embeddings['ATAC'], atac_limits), ('Predicted ATAC', predicted_umap, atac_limits)]):
        ax = fig.add_subplot(grid[0, i])
        for label in draw_order:
            selected = labels == label
            ax.scatter(xy[selected, 0], xy[selected, 1], s=1.05, c=COLORS[label], alpha=.72, linewidths=0, rasterized=True)
        ax.set_title(title, pad=3, fontproperties=font)
        ax.set_xlim(lim[:, 0])
        ax.set_ylim(lim[:, 1])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()
        outside = int(np.sum(np.any((xy < lim[0]) | (xy > lim[1]), axis=1)))
        assert outside == 0
        plot_checks.append({'panel': title, 'n_metacells': len(xy), 'outside_axis_limits': outside, 'x_limits': lim[:, 0].tolist(), 'y_limits': lim[:, 1].tolist()})
    legend_ax = fig.add_subplot(grid[0, 3])
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    legend_ax.set_axis_off()
    handles = [Line2D([0], [0], marker='o', ls='none', mfc=COLORS[label], mec='none', ms=4.4, label=DISPLAY[label]) for label in LABELS]
    legend_ax.legend(handles=handles, loc='center left', bbox_to_anchor=(-.015, .5), ncol=1, frameon=False, handletextpad=.35, labelspacing=.45, borderaxespad=0, prop=font)
    fig.subplots_adjust(left=.008, right=.998, top=.90, bottom=.035)
    for text in fig.findobj(Text):
        text.set_fontproperties(font)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            assert box.x0 >= -.1 and box.x1 <= fig.bbox.x1 + .1 and box.y0 >= -.1 and box.y1 <= fig.bbox.y1 + .1, text.get_text()
    out.mkdir(parents=True)
    final = out / 'final'
    final.mkdir()
    stem = final / 'human_cerebral_t_mapping_reference'
    fig.savefig(stem.with_suffix('.png'), dpi=450)
    fig.savefig(stem.with_suffix('.pdf'), dpi=450)
    plt.close(fig)
    np.savez_compressed(out / 't_mapping_predictions_coordinates.npz', paired_metacell_id=ids, day=days,
                        model_time=times, lineage_coarse=labels, lineage_display=np.array([DISPLAY[v] for v in labels]),
                        source_line=lines, original_random_split=split_labels, rna_pca30_normalized=x,
                        observed_atac_lsi12_normalized=y, predicted_atac_lsi12_normalized=prediction,
                        rna_umap=embeddings['RNA'], observed_atac_umap=embeddings['ATAC'], predicted_atac_umap=predicted_umap)
    audit = {'identity': {'paired_ids_and_metadata_index_exact': True, 'RNA_and_ATAC_UMAP_ids_order_days_lines_exact': True, 'normalized_RNA_equal_metadata_X_latent': True},
             'normalization': norm_audit, 'UMAP_ATAC_reference_input_max_abs': raw_error, 'UMAP_ATAC_embedding_float32_max_abs': embedding_error,
             'UMAP_embedding_CSV_float64_precision_note': 'Stored UMAPs were originally float32; casting CSV coordinates back to float32 recovers exact reducer.embedding_.',
             'test_equal_time_standardized_MSE': equal_time_mse, 'checkpoint_test_equal_time_standardized_MSE': expected_mse,
             'test_MSE_absolute_difference': abs(equal_time_mse - expected_mse), 'test_by_day': test_sanity,
             'test_sanity_scope': 'Reproduction of original random within-age/line held-out cell split only; figure includes all train/val/test cells and is not LOO.', 'panels': plot_checks}
    (out / 'input_and_forward_audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    for record in before:
        assert sha(record['path']) == record['sha256'], f"Changed input {record['path']}"
    manifest = {'status': 'created_pending_final_PDF_visual_QA', 'analysis': 'human cerebral frozen T RNA-to-ATAC atlas', 'n_metacells': len(x),
                'days': list(DAYS), 'model_time_rule': '(day - 4) / 10', 'model_time_points': [float((d - 4) / 10) for d in DAYS],
                'counts_by_day': {str(d): int(np.sum(days == d)) for d in DAYS}, 'label_column': 'lineage_coarse', 'label_counts': counts, 'label_display': DISPLAY,
                'label_colors': COLORS, 'label_semantics': 'Observed paired-RNA metacell labels carried unchanged to observed and predicted ATAC; no prediction reclassification and no clonal tracing.',
                'T_checkpoint': str(CHECKPOINT), 'T_sha256': EXPECTED_T_SHA, 'T_config': checkpoint['config'], 'T_metadata': checkpoint['meta'],
                'normalization': normalization, 'prediction_output_space': 'normalized ATAC LSI12; FiLM forward restores only its internal target standardization, with no second external rescaling',
                'UMAP_policy': 'Reuse existing Model RNA and Model ATAC coordinates. Apply native transform of the exact matching frozen observed-ATAC UMAP to predicted normalized LSI12.',
                'UMAP_model': str(UMAP_MODEL), 'UMAP_reference_n_neighbors': reducer.n_neighbors, 'UMAP_reference_min_dist': reducer.min_dist,
                'UMAP_random_state': reducer.random_state, 'UMAP_transform_seed': reducer.transform_seed, 'UMAP_metric': reducer.metric,
                'extra_knn_projection': False, 'new_UMAP_fit': False, 'new_T_training': False, 'new_trajectory': False, 'HPC_access': False,
                'RNA_axis_limits': rna_limits.tolist(), 'shared_ATAC_and_prediction_axis_limits': atac_limits.tolist(), 'axis_policy': 'Full min/max union with 4% range padding, no quantile clipping',
                'font': 'Arial', 'font_file': str(FONT), 'font_size_pt': 10, 'width_mm': 200, 'height_mm': 60,
                'scatter_size_pt2': 1.05, 'scatter_alpha': .72, 'draw_order': draw_order, 'batch_size': args.batch_size, 'device': 'cpu',
                'PDF_creation_marker': 'Executed once by root before authoring; not rerun by execution agent',
                'limitations': ['All-seven-time all-cell mapping display, not a held-out-time prediction.', 'A UMAP projection is for display and does not prove numerical accuracy or distribution recovery.', 'Original T random-cell train/validation/test splits are retained; all are combined in the figure.', 'Spatial proximity in UMAP is not an independent biological validation or evidence of clonal lineage.'],
                'inputs': before, 'all_input_hashes_unchanged_after_run': True, 'audit_file': str(out / 'input_and_forward_audit.json'),
                'outputs': [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(out.rglob('*')) if p.is_file()]}
    (out / 'creation_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    (out / 'README.md').write_text(f'''# Human cerebral frozen-T RNA-to-ATAC reference atlas

![RNA, ATAC and predicted ATAC atlas]({stem.with_suffix('.png')})

The three panels show the same 20,663 paired metacells from D4, D7, D9, D11,
D12, D18 and D21. RNA uses normalized PCA30; ATAC uses normalized LSI12.
The third panel is T(RNA, time) using the authoritative frozen FiLM map, with
time = (day - 4) / 10. All points retain the same observed RNA lineage_coarse
annotation: Early (7,308), Non-telencephalon (2,739), Telencephalon (10,581),
and Other (35). These are population annotations, not traced lineages.

## Frozen inputs and projection

The T checkpoint is `{CHECKPOINT}` (SHA-256 `{EXPECTED_T_SHA}`). Its original
70/15/15 within-age-and-cell-line random split, seed 42, is preserved. This
figure combines all split groups and all seven time points. It is not LOO.
The model's forward pass already returns normalized LSI12; no second
normalization or denormalization is applied.

Observed coordinates are reused from the existing Model RNA and Model ATAC
UMAPs after exact metacell ID, order, day, source-line and space checks.
The frozen ATAC reducer input matches normalized LSI12 exactly, and its
embedding matches the restored float32 CSV coordinates exactly. Predicted
ATAC is projected with this reducer's native transform (n_neighbors=30,
min_dist=0.25, fit and transform seeds=42); no UMAP is refitted and no extra
kNN regression/smoothing is introduced. ATAC and predicted ATAC have shared
full-union bounds, including all points with 4% padding. Nothing is clipped.

## Scope and limitations

This is a full-seven-time mapping visualization, not quantitative proof of
accuracy or external biological validation. Reference UMAP projection can
place imperfect high-dimensional predictions near the observed atlas.
Closeness or separation must be assessed quantitatively before UMAP in a
separate analysis. Labels are never reclassified from predictions.

An input/forward sanity check reproduces the checkpoint's original
equal-time test-cell standardized MSE: {equal_time_mse:.12f}, compared with
the stored {expected_mse:.12f}. This checks T/time/normalization consistency,
not a new held-out-time evaluation and not a metric inferred from the plot.

## Deliverables

- `final/human_cerebral_t_mapping_reference.pdf` and adjacent PNG.
- `t_mapping_predictions_coordinates.npz`: exact cell IDs, labels, day/time,
  split membership, normalized RNA/observed ATAC/predicted ATAC and all 2D coordinates.
- `input_and_forward_audit.json`: identity, normalization and original test-split checks.
- `creation_manifest.json` and final `analysis_manifest.json`: full hashes, settings and limitations.
- Final PDF rendering and typography checks accompany the final review.

Reproduce from `.`, into a new directory:

```bash
python comparison/human_cerebral/plot_human_cerebral_rna_atac_t_reference.py --output-dir results/human_cerebral_t_mapping_reference_reproduction
```

The script refuses an existing output directory. No model training,
trajectory generation, HPC access, or overwrite of existing results is used.
''')
    print(json.dumps({'output': str(out), 'status': manifest['status'], 'metacells': len(x), 'test_sanity_max_abs_difference': abs(equal_time_mse - expected_mse), 'projection': 'native frozen UMAP transform'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
