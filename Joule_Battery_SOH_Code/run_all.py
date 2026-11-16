import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = (
    '01_raw_data_processing_V2_2_FINAL.py',
    '02_feature_processing_V2_0_FINAL.py',
    '03_cross_domain_feature_selection_V3_0_FINAL.py',
    '04_cross_domain_ensemble_transfer_learning_V4_0_2.py',
    '05_model_interpretability_V5_0_2_CONDITIONAL_FINAL.py',
    '06_ablation_studies_V6_0_FINAL.py',
    '07_few_shot_cross_domain_transfer_V7_2_0_CONDITIONAL_FINAL.py',
)
OUTPUTS = (
    'DATAPROCESSING_V2_2', 'FEATUREPROCESSING_V2_0',
    'FEATURESELECTION_V3_0_1', 'SOH_RESULTS_ETL_V4_0_2_CONDITIONAL_CONTRIBUTION',
    'INTERPRETABILITY_V5_0_2_CONDITIONAL', 'ABLATION_RESULTS_V6_0_CONDITIONAL',
    'CROSS_DOMAIN_RESULTS_V7_2_0_CONDITIONAL',
)
FIGURES = (
    'figure1_dataset_overview.py', 'figure2_feature_space.py',
    'figure3_feature_selection.py', 'figure4_ensemble_transfer_performance.py',
    'figure5_interpretability.py', 'figure6_ablation.py',
    'figure7_few_shot_generalization.py',
)


def main():
    parser = argparse.ArgumentParser(description='Run manuscript experiments and figures.')
    parser.add_argument('--base-path', type=Path, default=Path(os.environ.get('CRPS_BASE_PATH', ROOT / 'workspace')))
    parser.add_argument('--from-stage', type=int, choices=range(1, 8), default=1)
    parser.add_argument('--through-stage', type=int, choices=range(1, 8), default=7)
    parser.add_argument('--figures-only', action='store_true')
    parser.add_argument('--skip-figures', action='store_true')
    parser.add_argument('--stage4-run', type=Path)
    args = parser.parse_args()
    if args.from_stage > args.through_stage:
        parser.error('--from-stage must not exceed --through-stage')
    if args.figures_only and args.skip_figures:
        parser.error('--figures-only and --skip-figures cannot be combined')
    if args.stage4_run and not args.stage4_run.is_dir():
        parser.error('--stage4-run must name an existing run directory')
    if args.stage4_run and not args.figures_only and args.from_stage <= 4 <= args.through_stage:
        parser.error('Do not select an existing Stage 4 run while generating a new Stage 4 run')
    base = args.base_path.expanduser().resolve()
    if not base.is_dir():
        parser.error('Data root does not exist. Place datasets in workspace/ or use --base-path PATH.')
    if not args.figures_only and args.from_stage == 1:
        paths = [base / 'Battery Dataset' / ('Batch-' + str(i)) for i in range(1, 7)]
        paths += [base / 'our_data', base / 'TJU/Dataset_2_NCM_battery', base / 'TJU/Dataset_1_NCA_battery']
        missing = [str(p) for p in paths if not p.is_dir()]
        if missing:
            parser.error('Missing raw data directories:\n' + '\n'.join(missing))
    env = dict(os.environ, CRPS_BASE_PATH=str(base), PYTHONUNBUFFERED='1')
    if args.stage4_run:
        env['CRPS_STAGE4_RUN'] = str(args.stage4_run.resolve())
    record = {'started_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
              'platform': ' '.join((platform.system(), platform.release(), platform.machine())), 'base_path': str(base), 'runs': {}, 'status': 'running', 'packages': {}}
    for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'torch', 'matplotlib', 'seaborn']:
        try:
            record['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            record['packages'][name] = None
    logs = base / 'execution_logs'
    logs.mkdir(exist_ok=True)
    log = logs / (datetime.now().strftime('execution_%Y%m%d_%H%M%S_%f') + '.json')
    try:
        if not args.figures_only:
            for number in range(args.from_stage, args.through_stage + 1):
                directory = base / OUTPUTS[number - 1]
                before = set(directory.glob('run_*'))
                command = [sys.executable, '-u', str(ROOT / STAGES[number - 1])]
                if number == 7:
                    command += ['--base-path', str(base)]
                    if env.get('CRPS_STAGE4_RUN'):
                        command += ['--stage4-run', env['CRPS_STAGE4_RUN']]
                print('Running Stage', number, flush=True)
                subprocess.run(command, env=env, check=True, cwd=ROOT)
                created = sorted(set(directory.glob('run_*')) - before)
                if len(created) == 1:
                    record['runs'][str(number)] = str(created[0])
                    if number == 4:
                        env['CRPS_STAGE4_RUN'] = str(created[0])
        if not args.skip_figures:
            numbers = range(1, 8) if args.figures_only else range(args.from_stage, args.through_stage + 1)
            for number in numbers:
                command = [sys.executable, '-u', str(ROOT / 'figures' / FIGURES[number - 1]), '--base-path', str(base)]
                selected = record['runs'].get(str(number))
                if number == 4 and not selected:
                    selected = env.get('CRPS_STAGE4_RUN')
                if selected:
                    command += ['--run', selected]
                subprocess.run(command, env=env, check=True, cwd=ROOT)
        record['status'] = 'completed'
    except BaseException as error:
        record['status'] = 'failed'
        record['error'] = str(error)
        raise
    finally:
        record['finished_utc'] = datetime.now(timezone.utc).isoformat()
        log.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        print('Execution record:', log, flush=True)


if __name__ == '__main__':
    main()
