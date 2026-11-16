import csv
import importlib.util
import tempfile
import unittest
import subprocess
import sys
import json
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from joule_extension import audit_manifest, lock_json, load, resolve_stage3, stage3_files, preflight

ROOT=Path(__file__).resolve().parent
repeat=load(ROOT/'09_repeat_source_pretraining.py','repeat_test')
report=load(ROOT/'10_summarize_repeats.py','report_test')

class Checks(unittest.TestCase):
    def make_run(self, root):
        run=root/'stage4'
        run.mkdir()
        report_dir=run/'validation_report'
        report_dir.mkdir()
        stage3=root/'FEATURESELECTION_V3_0_1'/'run_20260926_100000'
        (stage3/'datasets').mkdir(parents=True)
        for f in stage3_files(stage3):f.write_bytes(b'fixture')
        cfg={'stage3_run':r'C:\old_location\FEATURESELECTION_V3_0_1\run_20260926_100000',
            'script_build':'STAGE4_V4_0_2_CONDITIONAL_CONTRIBUTION_R2_20261006',
            'schema_version':'4.0.2','test_labels_used_for_training':False,
            'test_labels_used_for_early_stopping':False,'test_labels_used_for_ensemble_weights':False}
        (report_dir/'processing_config.json').write_text(json.dumps(cfg))
        (report_dir/'processing_summary.json').write_text(json.dumps({'validation_status':'PASS'}))
        (run/'source_standardizer.pkl').write_bytes(b'fixture')
        with (run/'battery_split_manifest.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['dataset','battery_id','split','sequence_eligible'])
            for d in ('TJU_NCM','TJU_NCA'):
                for i,role in enumerate(('target_adaptation_train','target_validation','target_test')):
                    w.writerow([d,str(i),role,'True'])
        return run,stage3,cfg

    def test_exact_relocation_no_latest_substitution(self):
        with tempfile.TemporaryDirectory() as t:
            run,s3,cfg=self.make_run(Path(t))
            self.assertEqual(resolve_stage3(cfg,Path(t)),s3.resolve())
            cfg['stage3_run']=r'C:\old\run_DIFFERENT'
            with self.assertRaises(FileNotFoundError):resolve_stage3(cfg,Path(t))

    def test_08_direct_no_arguments(self):
        audit=load(ROOT/'08_audit_existing_run.py','audit_test')
        with tempfile.TemporaryDirectory() as t:
            run,s3,cfg=self.make_run(Path(t))
            with patch.object(audit,'STAGE4_RUN',run),patch.object(audit,'DATA_ROOT',Path(t)),patch.object(sys,'argv',['08_audit_existing_run.py']):
                audit.main()

    def test_09_default_arguments_plan(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            run,s3,cfg=self.make_run(root)
            with patch.object(repeat,'STAGE4_RUN',run),patch.object(repeat,'DATA_ROOT',root),patch.object(repeat,'OUTPUT',root/'out'),patch.object(sys,'argv',['09_repeat_source_pretraining.py','--plan-only']):
                repeat.main()
            self.assertFalse((root/'out').exists())

    def test_10_direct_complete_synthetic_run(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            (root/'locked_protocol.json').write_text(json.dumps({'source_seeds':[101,211], 'target_seeds':[42], 'budgets':[.1]}))
            for source in [101,211]:
                run=root/('source_seed_'+str(source))
                (run/'validation_report').mkdir(parents=True)
                (run/'validation_report/processing_summary.json').write_text(json.dumps({'validation_status':'PASS'}))
                for case in ['XJTU_HUST_to_TJU_NCM','TJU_NCM_to_TJU_NCA']:
                    folder=run/'predictions'/case/'seed_42'/'budget_10'
                    folder.mkdir(parents=True)
                    for arm in ['transfer','scratch']:
                        f=pd.DataFrame({'dataset':['D']*3,'battery_id':['a','b','b'], 'window_index':[0,1,2], 'cycle':[20,20,21],'y_true':[.9,.8,.7]})
                        for n in report.NAMES+['Adaptive_Ensemble']:f[n]=f.y_true+(.01 if arm=='transfer' else .02)
                        f.to_csv(folder/(arm+'_immutable_final_test.csv'),index=False)
                        (folder/(arm+'_fusion_policy.json')).write_text(json.dumps({'selected_models':report.NAMES[:2]}))
            with patch.object(report,'OUTPUT',root),patch.object(sys,'argv',['10_summarize_repeats.py']):report.main()
            effects=pd.read_csv(root/'repeat_summary/paired_initialization_effects.csv')
            self.assertTrue(np.allclose(effects.scratch_minus_transfer,.01))

    def test_plan_without_torch_or_data(self):
        with tempfile.TemporaryDirectory() as t:
            run,_,_=self.make_run(Path(t))
            result=subprocess.run([sys.executable,str(ROOT/'09_repeat_source_pretraining.py'),
                '--base-path',t,'--stage4-run',str(run),'--output',str(Path(t)/'out'),
                '--plan-only'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('620',result.stdout)
    def test_seed_isolation(self):
        def original(*args,**kw): return args[5]
        m=SimpleNamespace(fit_arm=original)
        repeat.install_source_seed(m,101)
        self.assertEqual(m.fit_arm(None,None,None,None,None,42,'XJTU_HUST_to_TJU_NCM_source',source=True),101)
        self.assertEqual(m.fit_arm(None,None,None,None,None,42,'TJU_NCM_to_TJU_NCA_source',source=True),1000101)
        self.assertEqual(m.fit_arm(None,None,None,None,None,42,'target',source=False),42)
    def test_lock(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'lock.json'
            lock_json(path,{'a':1});lock_json(path,{'a':1})
            with self.assertRaises(ValueError):lock_json(path,{'a':2})
    def test_metrics_and_equal_fusion(self):
        f=pd.DataFrame({'dataset':['D']*3,'battery_id':['a','b','b'],'y_true':[0.,0.,0.]})
        for n in report.NAMES+['Adaptive_Ensemble']:f[n]=[1.,0.,0.]
        self.assertAlmostEqual(report.rmse(f,'Adaptive_Ensemble',False),np.sqrt(1/3))
        self.assertAlmostEqual(report.rmse(f,'Adaptive_Ensemble',True),.5)
        rows=list(report.comparisons(f,{'selected_models':report.NAMES[:2]}))
        self.assertEqual(len(rows),14)
    def test_duplicate_manifest(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'manifest.csv'
            with path.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['dataset','battery_id','split','sequence_eligible'])
                w.writerow(['D','a','source_train','True']);w.writerow(['D','a','target_test','True'])
            with self.assertRaises(ValueError):audit_manifest(path)

if __name__=='__main__':unittest.main()
