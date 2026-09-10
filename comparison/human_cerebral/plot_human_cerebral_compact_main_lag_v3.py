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
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np

ROOT = Path('.')
BASE = ROOT / 'results/human_cerebral_telencephalic_temporal_ranked_list_v1'
FONT = Path(__import__('matplotlib.font_manager', fromlist=['findfont']).findfont('Arial'))
PDF_PYTHON = _sys.executable
PAIR_IDS = ['DFPG01086510', 'DFPG01792220', 'DFPG01763210',
            'DFPG00561740', 'DFPG01792150', 'DFPG01086513']


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def inspect_pdf(path: Path) -> dict:
    code = '''import json, sys, pdfplumber
with pdfplumber.open(sys.argv[1]) as document:
    page = document.pages[0]
    print(json.dumps({"text": page.extract_text(), "lines": page.lines,
                      "width": page.width, "height": page.height,
                      "page_count": len(document.pages),
                      "all_arial10": all("Arial" in c["fontname"] and abs((c["size"] if c["upright"] else c["width"]) - 10) < 1e-7 for c in page.chars)}))
'''
    return json.loads(subprocess.check_output([PDF_PYTHON, '-c', code, str(path)], text=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=BASE / '08_main_compact_lag_v3/final')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(out)
    previous = BASE / '07_main_with_lag_v2/final'
    old_pdf = previous / 'main_temporal_ranked_2x3_with_lag.pdf'
    label_manifest = previous / 'creation_manifest.json'
    interface = BASE / '02_scores/interface_ready.json'
    cache = BASE / '02_scores/cache/primary_display_curves.npz'
    expected = {r['path']: r['sha256'] for r in json.loads(interface.read_text())['files']}
    assert sha(cache) == expected[str(cache)]
    metadata = json.loads(label_manifest.read_text())
    labels = metadata['labels']
    assert [r['lag_days'] for r in labels] == [3, 0, -3, 3, 0, -3]
    arrays = np.load(cache, allow_pickle=False)
    mapping = {p: i for i, p in enumerate(arrays['pair_ids'].astype(str))}
    ids = [mapping[p] for p in PAIR_IDS]
    values = arrays['dense_mean_temporal_z'][:, ids, :]
    days = arrays['days_dense']
    assert values.shape == (66, 6, 2)
    old = inspect_pdf(old_pdf)
    assert all(r['title'] in old['text'] for r in labels)
    spines = sorted([line for line in old['lines'] if line['width'] == 0 and 80 < line['height'] < 100], key=lambda line: (line['top'], line['x0']))
    assert len(spines) == 6
    lefts = [line['x0'] for line in spines[:3]]
    widths = [line['width'] for line in old['lines'] if line['height'] == 0 and 120 < line['width'] < 130]
    assert len(widths) == 6
    axis_width = widths[0]
    inputs = [old_pdf, label_manifest, interface, cache, FONT, Path(__file__).resolve()]
    records = [{'path': str(p), 'sha256': sha(p)} for p in inputs]
    fontManager.addfont(str(FONT))
    font = FontProperties(fname=str(FONT), size=10)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10, 'figure.labelsize': 10, 'pdf.fonttype': 42, 'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6, 'savefig.facecolor': 'white'})
    width, height = 190 * 72 / 25.4, 65 * 72 / 25.4
    fig = plt.figure(figsize=(190 / 25.4, 65 / 25.4))
    extent = max(2.8, 1.05 * float(np.abs(values).max()))
    panel_records = []
    for i, info in enumerate(labels):
        bottom = 104 if i < 3 else 30
        ax = fig.add_axes([lefts[i % 3] / width, bottom / height, axis_width / width, 46 / height])
        ax.set_title(info['title'], loc='left', pad=3)
        for m, (color, style) in enumerate([('#0072B2', '-'), ('#D55E00', '--')]):
            ax.plot(days, values[:, i, m], color=color, ls=style, lw=1.25)
        ax.text(.02, .98, info['label'], transform=ax.transAxes, ha='left', va='top', fontproperties=font)
        ax.set_xlim(4, 21)
        ax.set_ylim(-extent, extent)
        ax.set_xticks([4, 12, 21], ['D4', 'D12', 'D21'])
        ax.set_xticks([7, 9, 11, 18], minor=True)
        ax.set_yticks([-2, 0, 2])
        ax.tick_params(axis='x', which='major', labelbottom=True, length=3)
        ax.tick_params(axis='x', which='minor', length=2)
        ax.tick_params(axis='y', length=3, labelleft=(i % 3 == 0))
        panel_records.append({'pair_id': PAIR_IDS[i], **info, 'axis_left_points': lefts[i % 3], 'axis_bottom_points': bottom, 'axis_height_points': 46, 'axis_width_points': axis_width})
    legend = [Line2D([0], [0], color='#0072B2', lw=1.3, label='RNA'), Line2D([0], [0], color='#D55E00', lw=1.3, ls='--', label='ATAC')]
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.55, .995), ncol=2, frameon=False, handlelength=2, columnspacing=1.6, prop=font)
    fig.supylabel('Trajectory z-score', x=.028, fontsize=10)
    fig.supxlabel('Developmental day', y=.006, fontsize=10)
    for text in fig.findobj(Text):
        text.set_fontproperties(font)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            if box.x0 < -.1 or box.y0 < -.1 or box.x1 > fig.bbox.x1 + .1 or box.y1 > fig.bbox.y1 + .1:
                outside.append(text.get_text())
    assert not outside, outside
    out.mkdir(parents=True)
    pdf = out / 'main_temporal_ranked_2x3_compact_lag.pdf'
    fig.savefig(pdf)
    plt.close(fig)
    subprocess.run(['pdftoppm', '-singlefile', '-r', '180', '-png', str(pdf), str(pdf.with_suffix(''))], check=True)
    inspected = inspect_pdf(pdf)
    assert inspected['page_count'] == 1
    assert inspected['all_arial10']
    assert inspected['text'].count('lag') == 6
    assert all(r['title'] in inspected['text'] for r in labels)
    for record in records:
        assert sha(Path(record['path'])) == record['sha256']
    manifest = {'status': 'created_pending_visual_review', 'scope': 'main figure layout only', 'width_mm': 190, 'height_mm': 65, 'previous_height_mm': 115, 'height_ratio': 65 / 115, 'axis_row_gap_points': 28, 'font': 'Arial', 'font_size_pt': 10, 'source_cache_values_used_directly': True, 'no_new_scoring_or_normalization': True, 'lag_semantics_unchanged': True, 'x_axis_label_unchanged': 'Developmental day', 'y_limits': [-extent, extent], 'panels': panel_records, 'inputs': records, 'inputs_unchanged': True, 'outside_text_count': 0, 'pdf_skill_authoring_marker_executed_once': True, 'outputs': [{'path': str(p), 'sha256': sha(p)} for p in [pdf, pdf.with_suffix('.png')]]}
    (out / 'creation_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'pdf': str(pdf), 'png': str(pdf.with_suffix('.png')), 'dimensions_mm': [190, 65]}))


if __name__ == '__main__':
    main()
