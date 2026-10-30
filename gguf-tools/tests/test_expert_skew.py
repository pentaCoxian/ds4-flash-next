"""expert_skew: coverage, entropy and the gate on synthetic hotlists."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'mixed'))
import expert_skew as skew


def write_hotlist(path, counts):
    lines = ['# ds4 expert hotlist v1', '# layers 1', f'# experts {len(counts)}',
             '# columns: layer expert hits weight']
    lines += [f'0 {e} {c} {c * 0.1}' for e, c in enumerate(counts)]
    path.write_text('\n'.join(lines) + '\n')


class ExpertSkew(unittest.TestCase):
    def run_main(self, *files):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(skew.main([f'w{i}={f}' for i, f in enumerate(files)]), 0)
        return out.getvalue()

    def test_uniform_routing_is_not_worth_a_split(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d)/'u.txt'
            write_hotlist(f, [10] * 512)
            out = self.run_main(f, f)
        self.assertIn('median weighted share of the top 128 experts: 0.250', out)
        self.assertIn('not worth building', out)

    def test_concentrated_shared_routing_is(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d)/'s.txt'
            write_hotlist(f, [1000] * 64 + [1] * 448)
            out = self.run_main(f, f)
        self.assertIn('lowest cross-workload Jaccard of the top-128 sets: 1.000', out)
        self.assertIn('per-expert precision split: worth building', out)

    def test_entropy_and_cover(self):
        self.assertAlmostEqual(skew.entropy([1.0] * 8), 1.0)
        self.assertAlmostEqual(skew.cover([4, 3, 2, 1], 2), 0.7)
        self.assertAlmostEqual(skew.spearman([1, 2, 3], [10, 20, 30]), 1.0)


if __name__ == '__main__':
    unittest.main()
