import argparse
import json
import os
import platform
from pathlib import Path
from joule_extension import S4, S7, audit_manifest, digest, load, lock_json, preflight
from joule_paths import DATA_ROOT, STAGE4_RUN, OUTPUT

def install_source_seed(module, source_seed):
    original = module.fit_arm
    def fit(stage4, folder, train, validation, source_states, seed, phase, source=False):
        if source:

            seed = source_seed + (1000000 if phase.startswith('TJU_NCM_to_TJU_NCA') else 0)
        return original(stage4, folder, train, validation, source_states, seed, phase, source=source)
    module.fit_arm = fit

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-path', default=DATA_ROOT, type=Path)
    p.add_argument('--stage4-run', default=STAGE4_RUN, type=Path)
    p.add_argument('--output', default=OUTPUT, type=Path)
    p.add_argument('--source-seeds', nargs='+', type=int, default=[101, 211, 307, 401, 503])
    p.add_argument('--target-seeds', nargs='+', type=int, default=[42, 52, 62])
    p.add_argument('--budgets', nargs='+', type=float, default=[0.1, 0.2, 0.4, 0.6, 0.8])
    p.add_argument('--bootstrap-iterations', type=int, default=1000)
    p.add_argument('--plan-only', action='store_true')
    a = p.parse_args()
    if a.stage4_run is None:
        p.error("Specify --stage4-run or set CRPS_STAGE4_RUN to an exact accepted Stage 4 run")
    if len(set(a.source_seeds)) != len(a.source_seeds) or len(set(a.target_seeds)) != len(a.target_seeds):
        p.error('Duplicate seed')
    if any(s < 0 or s >= 1000000 for s in a.source_seeds + a.target_seeds):
        p.error('Seeds must be between 0 and 999999')
    if not a.budgets or any(b <= 0 or b > .8 for b in a.budgets) or len(set(a.budgets)) != len(a.budgets):
        p.error('Unique positive budgets <= 0.8 required')
    if any(abs(100*b-round(100*b)) > 1e-8 for b in a.budgets):
        p.error('Budgets must be integer percentages (original checkpoint naming)')
    folder = Path(__file__).resolve().parent
    stage4 = folder/S4
    stage7 = folder/S7
    manifest = a.stage4_run/'battery_split_manifest.csv'
    print('正在检查数据和历史结果路径，请稍候……', flush=True)
    _, stage3_run, counts, data_hashes = preflight(a.stage4_run, a.base_path, folder)
    protocol = {'extension': 'source-repeat-v2-direct', 'stage4_run': str(a.stage4_run.resolve()),
        'resolved_stage3_run': str(stage3_run.resolve()), 'input_sha256': data_hashes,
        'base_path': str(a.base_path.resolve()), 'source_seeds': a.source_seeds,
        'target_seeds': a.target_seeds, 'budgets': sorted(a.budgets),
        'bootstrap_iterations': a.bootstrap_iterations,
        'sha256': {x.name: digest(x) for x in (stage4, stage7, manifest, Path(__file__), folder/'joule_extension.py', folder/'joule_paths.py')},
        'historical_test_independence': False, 'evaluation': 'retrospective fixed benchmark',
        'source_preprocessing_shared_by_arms': True}
    print(json.dumps({'protocol': protocol, 'manifest_counts': counts,
        'neural_fits_estimate': len(a.source_seeds)*2*(2+len(a.target_seeds)*len(a.budgets)*4)}, indent=2))
    if a.plan_only:
        return

    os.environ['CRPS_BASE_PATH'] = str(a.base_path.resolve())
    print('\n路径检查完成。开始补充实验（支持同配置断点续跑）。', flush=True)
    print('输出目录：', a.output, flush=True)
    lock_json(a.output/'locked_protocol.json', protocol)
    for seed in a.source_seeds:
        module = load(stage7, 'joule_stage7_source_' + str(seed))
        module.SEEDS = tuple(a.target_seeds)
        module.BUDGETS = (0.,) + tuple(sorted(a.budgets))
        module.BUILD = module.BUILD + '_SOURCE_REPEAT_' + str(seed)

        original_accepted = module.accepted_stage4
        def accepted(root, explicit, original=original_accepted):
            run_path, config = original(root, explicit)
            config = dict(config, stage3_run=str(stage3_run))
            return run_path, config
        module.accepted_stage4 = accepted
        install_source_seed(module, seed)
        run = a.output/('source_seed_' + str(seed))
        module.args_parser = lambda: argparse.Namespace(base_path=a.base_path,
            stage4_run=a.stage4_run, stage4_script=stage4, run_dir=run,
            bootstrap_iterations=a.bootstrap_iterations)
        import numpy, pandas, scipy, sklearn, torch
        lock_json(run/'environment.json', {'python': platform.python_version(),
            'numpy': numpy.__version__, 'pandas': pandas.__version__, 'scipy': scipy.__version__,
            'sklearn': sklearn.__version__, 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})
        module.main()
    print('Source repeats complete. Run 10_summarize_repeats.py --root with the same output directory.')

if __name__ == '__main__':
    main()
