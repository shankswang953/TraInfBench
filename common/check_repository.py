#!/usr/bin/env python3
"""Check the lightweight repository without loading data or training models."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    errors: list[str] = []
    files = [p for base in ('model', 'comparison', 'common') for p in (ROOT / base).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts]
    sources = [p for p in files if p.suffix in ('.py', '.sh', '.R')]
    modules = {p.stem for p in sources if p.suffix == '.py'}
    for path in sources:
        text = path.read_text()
        if path.suffix == '.py':
            try:
                ast.parse(text, filename=str(path))
            except SyntaxError as exc:
                errors.append(str(exc))
        elif path.suffix == '.sh':
            result = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
            if result.returncode:
                errors.append(result.stderr)
        if re.search(r'[\"\']/Users/[^\n\"\']+', text):
            errors.append(f'Nonportable user path: {path.relative_to(ROOT)}')
    manifest = json.loads((ROOT / 'common/source_manifest.json').read_text())
    for record in manifest:
        if not (ROOT / record['path']).is_file():
            errors.append(f'Missing selected source: {record["path"]}')
        for dep in record.get('dependencies', []):
            if not (ROOT / dep).is_file():
                errors.append(f'Missing helper: {record["path"]} -> {dep}')
    figures = json.loads((ROOT / 'comparison/figure_manifest.json').read_text())['figures']
    for figure in figures:
        for target in figure['scripts']:
            if not (ROOT / target).is_file():
                errors.append(f'Missing figure source: {figure["figure"]} -> {target}')
    for path in files:
        if path.stat().st_size > 1024 * 1024:
            errors.append(f'Unexpected large release file: {path.relative_to(ROOT)}')
        if path.suffix in {'.h5ad', '.npz', '.npy', '.pt', '.pth', '.joblib', '.pkl', '.pdf', '.png'}:
            errors.append(f'Generated/data artifact in source tree: {path.relative_to(ROOT)}')
    for method in ('TrajectoryNet', 'MIOFlow', 'CytoBridge', 'TIGON', 'scMultiNODE'):
        if not (ROOT / 'model' / method / 'README.md').is_file():
            errors.append(f'Missing method guide: {method}')
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        raise SystemExit(1)
    print(f'PASS: {len(sources)} source files, {len(figures)} figure groups; syntax, paths, manifests and size checks.')
    print('No training performed. Exact figure replay requires the documented external assets.')


if __name__ == '__main__':
    main()
