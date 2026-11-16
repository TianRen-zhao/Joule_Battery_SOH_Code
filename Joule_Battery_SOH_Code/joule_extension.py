import hashlib
import importlib.util
import json
import sys
from pathlib import Path, PureWindowsPath

S4 = '04_cross_domain_ensemble_transfer_learning_V4_0_2.py'
S7 = '07_few_shot_cross_domain_transfer_V7_2_0_CONDITIONAL_FINAL.py'

def stage3_files(run):
    return [run/'selected_features.txt'] + [
        run/'datasets'/(d+'_selected_features.pkl')
        for d in ('XJTU', 'HUST', 'TJU_NCM', 'TJU_NCA')]

def resolve_stage3(config, base):
    if not config.get('stage3_run'):
        raise ValueError('Stage 4 配置缺少 stage3_run，无法确认数据来源。')
    text = str(config['stage3_run'])
    old = Path(text)
    name = PureWindowsPath(text).name if '\\' in text else old.name
    if all(p.is_file() for p in stage3_files(old)):
        return old
    root = Path(base)/'FEATURESELECTION_V3_0_1'
    candidates = [root/name]
    if root.is_dir():
        candidates += [p for p in root.glob('*/'+name) if p.is_dir()]
    valid = sorted(set(p.resolve() for p in candidates if all(f.is_file() for f in stage3_files(p))))
    if len(valid) == 1:
        print('已找到原 Stage 3 同名运行目录：', valid[0], flush=True)
        return valid[0]
    if len(valid) > 1:
        raise ValueError('找到多个同名 Stage 3 数据目录，不能自动选择：\n'+'\n'.join(map(str,valid)))
    raise FileNotFoundError('缺少原实验的 Stage 3 特征数据。原配置路径：\n'+text+
        '\n也检查了：\n'+str(root/name)+
        '\n请恢复该目录中的 selected_features.txt 和 datasets 下四个 selected_features.pkl 文件。无需重跑 01–07。')

def preflight(stage4_run, base, folder):
    stage4_run, folder = Path(stage4_run), Path(folder)
    required = [stage4_run/'battery_split_manifest.csv', stage4_run/'source_standardizer.pkl',
        stage4_run/'validation_report/processing_config.json',
        stage4_run/'validation_report/processing_summary.json', folder/S4, folder/S7]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError('找不到以下必要文件：\n'+'\n'.join(missing))
    config = json.loads(required[2].read_text(encoding='utf-8-sig'))
    summary = json.loads(required[3].read_text(encoding='utf-8-sig'))
    if config.get('script_build') != 'STAGE4_V4_0_2_CONDITIONAL_CONTRIBUTION_R2_20261006' or config.get('schema_version') != '4.0.2':
        raise ValueError('Stage 4 版本不匹配，请保留原始 04 文件及对应运行结果。')
    if summary.get('validation_status') != 'PASS':
        raise ValueError('原 Stage 4 检查未通过，停止补充训练。')
    for key in ('test_labels_used_for_training','test_labels_used_for_early_stopping','test_labels_used_for_ensemble_weights'):
        if config.get(key) is not False:
            raise ValueError('原配置缺少或未通过检查：'+key)
    counts = audit_manifest(required[0])
    stage3 = resolve_stage3(config, base)
    provenance = {str(p.resolve()): digest(p) for p in required[:4]+stage3_files(stage3)}
    return config, stage3, counts, provenance

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def lock_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError('Protocol changed; choose a NEW output directory: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2)

def audit_manifest(path):
    import csv
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    required = {'dataset', 'battery_id', 'split', 'sequence_eligible'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError('Missing manifest fields or empty manifest')
    seen, counts = set(), {}
    for row in rows:
        key = (row['dataset'], row['battery_id'])
        if key in seen:
            raise ValueError('Duplicate battery assignment: ' + repr(key))
        seen.add(key)
        k = row['dataset'] + '/' + row['split']
        counts[k] = counts.get(k, 0) + 1
        if row.get('used_for_scaler', '').lower() in ('true', '1') and row['split'] != 'source_train':
            raise ValueError('Scaler assignment outside source_train')
    for d in ('TJU_NCM', 'TJU_NCA'):
        for role in ('target_adaptation_train', 'target_validation', 'target_test'):
            if not counts.get(d + '/' + role):
                raise ValueError('Missing target role: ' + d + '/' + role)
    return counts
