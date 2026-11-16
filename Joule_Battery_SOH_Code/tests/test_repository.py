import ast
import importlib.util
import io
import json
import sys
import tempfile
import tokenize
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_file(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RepositoryTests(unittest.TestCase):
    def test_python_files_parse_without_comments_or_docstrings(self):
        for path in ROOT.rglob('*.py'):
            if 'workspace' in path.parts or '.venv' in path.parts:
                continue
            source = path.read_text(encoding='utf-8-sig')
            tree = ast.parse(source)
            self.assertFalse(any(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(source).readline)), str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    self.assertIsNone(ast.get_docstring(node), str(path))

    def test_registered_scripts_exist(self):
        runner = load_file('run_all.py', 'release_runner')
        for filename in runner.STAGES:
            self.assertTrue((ROOT / filename).is_file(), filename)
        for filename in runner.FIGURES:
            self.assertTrue((ROOT / 'figures' / filename).is_file(), filename)

    def test_no_author_machine_paths(self):
        for path in ROOT.rglob('*.py'):
            if 'workspace' not in path.parts:
                self.assertNotIn('82601', path.read_text(encoding='utf-8'))

    def test_stage4_path_is_propagated_to_stage7(self):
        runner = load_file('run_all.py', 'propagation_runner')
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            run = base / runner.OUTPUTS[3] / 'run_fixture'
            run.mkdir(parents=True)
            argv = ['run_all.py', '--base-path', str(base), '--from-stage', '7', '--through-stage', '7', '--skip-figures', '--stage4-run', str(run)]
            with patch.object(sys, 'argv', argv), patch.object(runner.subprocess, 'run') as call:
                runner.main()
            command = call.call_args.args[0]
            self.assertEqual(command[command.index('--stage4-run') + 1], str(run))
            record = json.loads(next((base / 'execution_logs').glob('*.json')).read_text())
            self.assertEqual(record['status'], 'completed')

    def test_partial_run_only_draws_matching_figures(self):
        runner = load_file('run_all.py', 'partial_runner')
        with tempfile.TemporaryDirectory() as temporary:
            argv = ['run_all.py', '--base-path', temporary, '--from-stage', '2', '--through-stage', '3']
            with patch.object(sys, 'argv', argv), patch.object(runner.subprocess, 'run') as call:
                runner.main()
            commands = [c.args[0] for c in call.call_args_list]
            self.assertEqual(len(commands), 4)
            self.assertEqual(Path(commands[2][2]).name, runner.FIGURES[1])
            self.assertEqual(Path(commands[3][2]).name, runner.FIGURES[2])

    def test_missing_raw_inputs_fail_before_training(self):
        runner = load_file('run_all.py', 'missing_runner')
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(sys, 'argv', ['run_all.py', '--base-path', temporary]), patch.object(runner.subprocess, 'run') as call:
                with self.assertRaises(SystemExit) as result:
                    runner.main()
                self.assertEqual(result.exception.code, 2)
                call.assert_not_called()

    def test_failed_subprocess_is_recorded_and_stops(self):
        runner = load_file('run_all.py', 'failed_runner')
        with tempfile.TemporaryDirectory() as temporary:
            argv = ['run_all.py', '--base-path', temporary, '--from-stage', '2', '--skip-figures']
            error = runner.subprocess.CalledProcessError(1, 'fixture')
            with patch.object(sys, 'argv', argv), patch.object(runner.subprocess, 'run', side_effect=error) as call:
                with self.assertRaises(runner.subprocess.CalledProcessError):
                    runner.main()
                self.assertEqual(call.call_count, 1)
            record = json.loads(next((Path(temporary) / 'execution_logs').glob('*.json')).read_text())
            self.assertEqual(record['status'], 'failed')

    def test_rmse_definitions(self):
        import numpy as np
        import pandas as pd
        summary = load_file('10_summarize_repeats.py', 'metrics_summary')
        frame = pd.DataFrame({'dataset': ['D'] * 3, 'battery_id': ['a', 'b', 'b'], 'y_true': [0.0] * 3, 'prediction': [1.0, 2.0, 2.0]})
        self.assertAlmostEqual(summary.rmse(frame, 'prediction', False), np.sqrt(3))
        self.assertAlmostEqual(summary.rmse(frame, 'prediction', True), 1.5)


if __name__ == '__main__':
    unittest.main()
