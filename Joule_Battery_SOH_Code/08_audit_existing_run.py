import argparse
import json
from pathlib import Path
from joule_extension import audit_manifest, digest, preflight
from joule_paths import DATA_ROOT, STAGE4_RUN

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage4-run', default=STAGE4_RUN, type=Path)
    p.add_argument('--base-path', default=DATA_ROOT, type=Path)
    a = p.parse_args()
    if a.stage4_run is None:
        p.error("Specify --stage4-run or set CRPS_STAGE4_RUN to an exact accepted Stage 4 run")
    run = a.stage4_run
    _, stage3, _, _ = preflight(run, a.base_path, Path(__file__).resolve().parent)
    config = json.loads((run/'validation_report/processing_config.json').read_text(encoding='utf-8-sig'))
    summary = json.loads((run/'validation_report/processing_summary.json').read_text(encoding='utf-8-sig'))
    for key in ('test_labels_used_for_training', 'test_labels_used_for_early_stopping', 'test_labels_used_for_ensemble_weights'):
        if config.get(key) is not False:
            raise ValueError('Missing/unsafe provenance flag: ' + key)
    if summary.get('validation_status') != 'PASS':
        raise ValueError('Existing Stage 4 integrity checks did not pass')
    manifest = run/'battery_split_manifest.csv'
    print(json.dumps({'manifest_counts': audit_manifest(manifest),
        'manifest_sha256': digest(manifest),
        'stage4_integrity_status': summary['validation_status'],
        'historical_test_independence': 'NOT ESTABLISHED by this audit',
        'nested_pipeline_validation': False,
        'note': 'Checks saved assignments/claims; does not audit raw windows or prove absence of human test feedback.'}, indent=2))
    print('\n08 检查通过。已找到原特征数据：', stage3)
    print('Optional next step: run 09_repeat_source_pretraining.py with the same base path and Stage 4 run.')

if __name__ == '__main__':
    main()
