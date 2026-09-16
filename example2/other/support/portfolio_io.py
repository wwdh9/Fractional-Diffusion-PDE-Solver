"""Organize completed outputs without changing any numerical calculation."""
from pathlib import Path
import csv
import json
import math
import shutil


ALPHAS = (0.4, 0.8, 1.2, 1.6, 2.0)
BRANCHES = ('u1', 'u2', 'total')
METRICS = ('mae', 'rmse', 'max_abs', 'relative_l2')
CSV_FIELDS = ['example', 'alpha', 'status'] + [
    f"{branch}_{'t1_' if scope == 't1' else ''}{metric}"
    for scope in ('all_times', 't1')
    for branch in BRANCHES
    for metric in METRICS
] + ['metrics_file']


def alpha_folder(alpha):
    return 'alpha_' + f'{alpha:.1f}'.replace('.', '_')


def is_training_image(name):
    return 'loss' in name.lower() or name == 'fno_raw_errors_icbc.png'


def summary_rows(version):
    version = Path(version)
    example = int(version.name.removeprefix('example'))
    rows = []
    for alpha in ALPHAS:
        relative = Path('other') / alpha_folder(alpha) / 'metrics.json'
        path = version / relative
        row = dict(example=example, alpha=f'{alpha:.1f}', status='not_run')
        if path.is_file():
            m = json.loads(path.read_text(encoding='utf-8-sig'))
            actual_alpha = m.get('alpha', m.get('config', {}).get('alpha'))
            if actual_alpha is None or abs(float(actual_alpha) - alpha) > 1e-12:
                raise ValueError(f'Alpha does not match directory: {path}')
            for scope in ('all_times', 't1'):
                for branch in BRANCHES:
                    for metric in METRICS:
                        value = float(m[scope][branch][metric])
                        if not math.isfinite(value):
                            raise ValueError(f'Nonfinite {scope}/{branch}/{metric}: {path}')
                        key = f"{branch}_{'t1_' if scope == 't1' else ''}{metric}"
                        row[key] = value
            row.update(status='completed', metrics_file=relative.as_posix())
        rows.append(row)
    return rows


def write_summary(version):
    version = Path(version)
    rows = summary_rows(version)
    if rows:
        with (version / 'results_summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)


def finalize(version, output):
    version, output = Path(version).resolve(), Path(output).resolve()
    # Custom --out results remain self-contained in the user-selected location.
    standard = output.parent == version / 'other' and output.name.startswith('alpha_')
    image = version / 'images' / output.name if standard else output / 'images'
    for source in output.glob('*.png'):
        destination = (output / 'training_image' if is_training_image(source.name) else image) / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
    if standard:
        write_summary(version)
    print(f'Result images: {image}; supporting files: {output}', flush=True)
