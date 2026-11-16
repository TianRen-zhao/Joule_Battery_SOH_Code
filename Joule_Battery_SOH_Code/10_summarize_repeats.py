import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from joule_paths import OUTPUT

NAMES = ['CNN_BiLSTM_Attention', 'MLP_Feature_Attention', 'HGB_Window_SOH', 'Ridge_Window_SOH']

def rmse(frame, column, macro):
    err = (frame[column].to_numpy(float)-frame.y_true.to_numpy(float))**2
    if not np.isfinite(err).all():
        raise ValueError('Nonfinite predictions')
    if macro:
        return float(np.mean([np.sqrt(err[ix].mean()) for ix in frame.groupby(['dataset','battery_id']).indices.values()]))
    return float(np.sqrt(err.mean()))

def comparisons(frame, policy):
    selected = policy['selected_models']
    if not selected or not set(selected).issubset(NAMES):
        raise ValueError('Unknown selected models')
    f = frame.copy()
    f['Equal_All'] = f[NAMES].mean(axis=1)
    f['Equal_Selected'] = f[selected].mean(axis=1)
    for c in NAMES + ['Equal_All', 'Equal_Selected', 'Adaptive_Ensemble']:
        for macro in (False, True):
            yield c, ('battery_macro' if macro else 'pooled_window'), rmse(f,c,macro)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default=OUTPUT, type=Path)
    a = p.parse_args()
    if not (a.root/'locked_protocol.json').is_file():
        raise FileNotFoundError('尚未找到 09 的实验记录。请先右键运行 09_repeat_source_pretraining.py。\n预期结果目录：'+str(a.root))
    protocol = json.loads((a.root/'locked_protocol.json').read_text(encoding='utf-8'))
    rows, expected = [], set()
    for s in protocol['source_seeds']:
        run = a.root/('source_seed_'+str(s))
        summary_path = run/'validation_report/processing_summary.json'
        if not summary_path.is_file():
            raise FileNotFoundError('09 尚未全部完成。请重新运行 09 续跑，完成后再运行 10。\n缺少：'+str(summary_path))
        summary = json.loads(summary_path.read_text(encoding='utf-8-sig'))
        if summary['validation_status'] != 'PASS':
            raise ValueError('Failed run: ' + str(run))
        for case in ['XJTU_HUST_to_TJU_NCM', 'TJU_NCM_to_TJU_NCA']:
            for t in protocol['target_seeds']:
                for b in protocol['budgets']:
                    frames = {}
                    for arm in ['transfer','scratch']:
                        folder = run/'predictions'/case/('seed_%d'%t)/('budget_%02d'%round(100*b))
                        f = pd.read_csv(folder/(arm+'_immutable_final_test.csv'), dtype={'battery_id':str})
                        keys=['dataset','battery_id','window_index','cycle']
                        if f.duplicated(keys).any():
                            raise ValueError('Duplicate prediction window')
                        f = f.sort_values(keys).reset_index(drop=True)
                        frames[arm] = f
                        policy = json.loads((folder/(arm+'_fusion_policy.json')).read_text(encoding='utf-8-sig'))
                        for model, scope, value in comparisons(f,policy):
                            rows.append(dict(source_seed=s,target_seed=t,case=case,budget=b,arm=arm,model=model,scope=scope,RMSE=value))
                    if not frames['transfer'][keys].equals(frames['scratch'][keys]) or not np.array_equal(frames['transfer'].y_true,frames['scratch'].y_true):
                        raise ValueError('Unmatched transfer/scratch windows or truth')
                    expected.add((s,t,case,b))
    out=a.root/'repeat_summary'
    out.mkdir(exist_ok=True)
    df=pd.DataFrame(rows)
    df.to_csv(out/'all_metrics.csv',index=False)
    df.groupby(['case','budget','arm','model','scope']).RMSE.agg(['mean','std','min','max','count']).to_csv(out/'descriptive_metrics.csv')
    key=['source_seed','target_seed','case','budget','model','scope']
    paired=df.pivot(index=key,columns='arm',values='RMSE').reset_index()
    paired['scratch_minus_transfer']=paired['scratch']-paired['transfer']
    paired.to_csv(out/'paired_initialization_effects.csv',index=False)
    fusion=df.pivot(index=['source_seed','target_seed','case','budget','arm','scope'],columns='model',values='RMSE').reset_index()
    for model in NAMES+['Equal_All','Equal_Selected']:
        fusion[model+'_minus_adaptive']=fusion[model]-fusion.Adaptive_Ensemble
    fusion.to_csv(out/'paired_fusion_effects.csv',index=False)
    rng=np.random.default_rng(8361)
    effects=[]
    for key,g in paired.groupby(['case','budget','model','scope']):
        v=g.groupby('source_seed').scratch_minus_transfer.mean().to_numpy()
        boot=rng.choice(v,(10000,len(v)),replace=True).mean(axis=1)
        effects.append(dict(zip(['case','budget','model','scope'],key),source_repeat_count=len(v),
            mean_effect=v.mean(),conditional_ci_low=np.quantile(boot,.025) if len(v)>1 else np.nan,
            conditional_ci_high=np.quantile(boot,.975) if len(v)>1 else np.nan))
    pd.DataFrame(effects).to_csv(out/'conditional_source_repeat_intervals.csv',index=False)
    print('Saved',out,'; matched experiment cells:',len(expected))
    print('Intervals condition on the historical test set; no multiplicity correction. No best-run selection.')

if __name__=='__main__':
    main()
