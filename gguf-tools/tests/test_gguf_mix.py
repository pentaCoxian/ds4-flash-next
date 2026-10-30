"""gguf_mix: the type table matches ds4, list reports the directory, and a
splice copies the donor's expert tensors byte for byte."""
import io
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'gguf-tools'/'mixed'))
import gguf_mix as mix

ALIGN = 32


def w_str(s):
    raw = s.encode()
    return struct.pack('<Q', len(raw)) + raw


def write_gguf(path, tensors):
    """tensors: [(name, dims, type, payload)] in file order."""
    head = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), 1)
    head += w_str('general.alignment') + struct.pack('<II', 4, ALIGN)
    off = 0
    infos = b''
    for name, dims, ggml_type, payload in tensors:
        infos += w_str(name) + struct.pack('<I', len(dims))
        infos += b''.join(struct.pack('<Q', d) for d in dims)
        infos += struct.pack('<IQ', ggml_type, off)
        off += mix.pad_to(len(payload), ALIGN)
    data = b''
    for _name, _dims, _type, payload in tensors:
        data += payload + b'\0' * (mix.pad_to(len(payload), ALIGN) - len(payload))
    blob = head + infos
    path.write_bytes(blob + b'\0' * (mix.pad_to(len(blob), ALIGN) - len(blob)) + data)


class TypeTable(unittest.TestCase):
    def test_matches_ds4_quants(self):
        src = (ROOT/'ds4_quants.c').read_text()
        ids = {m[1]: int(m[2]) for m in re.finditer(r'\bQ_(\w+)\s*=\s*(\d+)', src)}
        infos = re.findall(r'\{\s*Q_(\w+),\s*(\d+),\s*(\d+),', src)
        self.assertGreater(len(infos), 10)
        for name, block, n_bytes in infos:
            self.assertEqual(mix.GGML_QUANT_SIZES[ids[name]], (int(block), int(n_bytes), name))
        for name, n_bytes in (('F32', 4), ('F16', 2), ('BF16', 2)):
            self.assertEqual(mix.GGML_QUANT_SIZES[ids[name]], (1, n_bytes, name))


class ListAndSplice(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.base, self.donor, self.out = root/'base.gguf', root/'donor.gguf', root/'out.gguf'
        dims = (256, 2, 2)                       # 1024 weights per expert tensor
        q2 = lambda tag: bytes((tag + i) % 251 for i in range(1024 // 64 * 18))
        iq3 = lambda tag: bytes((tag * 7 + i) % 241 for i in range(1024 // 256 * 98))
        norm = struct.pack('<4f', 1, 2, 3, 4)
        self.base_t, self.donor_t = [], []
        for il in range(2):
            for k, part in enumerate(('gate', 'up', 'down')):
                name = f'blk.{il}.ffn_{part}_exps.weight'
                self.base_t.append((name, dims, 42, q2(il * 3 + k)))
                self.donor_t.append((name, dims, 18, iq3(il * 3 + k)))
        for t in (self.base_t, self.donor_t):
            t.insert(3, ('blk.1.attn_norm.weight', (4,), 0, norm))
        write_gguf(self.base, self.base_t)
        write_gguf(self.donor, self.donor_t)

    def test_list(self):
        out = io.StringIO()
        mix.list_tensors(mix.parse_gguf(self.base), out)
        rows = [l.split('\t') for l in out.getvalue().splitlines() if not l.startswith('#')]
        self.assertEqual([r[0] for r in rows], [t[0] for t in self.base_t])
        self.assertEqual(rows[0][1:3], ['Q2_0', '256,2,2'])
        self.assertEqual(rows[3][1:3], ['F32', '4'])
        data = self.base.read_bytes()
        for row, (_name, _dims, _type, payload) in zip(rows, self.base_t):
            off, n = int(row[3]), int(row[4])
            self.assertEqual(data[off:off + n], payload)
        self.assertIn('# Q2_0\t1728\t', out.getvalue())

    def test_splice_takes_donor_layer_bytes(self):
        subprocess.run([sys.executable, str(ROOT/'gguf-tools'/'mixed'/'gguf_mix.py'), 'splice',
                        '--base', str(self.base), '--donor', str(self.donor),
                        '--q4-layers', '1', '--out', str(self.out)],
                       check=True, capture_output=True)
        info = mix.parse_gguf(self.out)
        data = self.out.read_bytes()
        want = {name: (gtype, payload) for name, _dims, gtype, payload in self.base_t}
        for name, _dims, gtype, payload in self.donor_t:
            if name.startswith('blk.1.ffn_'):
                want[name] = (gtype, payload)
        for t in info.tensors:
            gtype, payload = want[t.name]
            self.assertEqual(t.ggml_type, gtype, t.name)
            self.assertEqual(data[t.data_offset:t.data_offset + t.n_bytes], payload, t.name)

    def test_splice_tensors_regex_and_provenance(self):
        subprocess.run([sys.executable, str(ROOT/'gguf-tools'/'mixed'/'gguf_mix.py'), 'splice',
                        '--base', str(self.base), '--donor', str(self.donor),
                        '--tensors', r'blk\.0\.ffn_down_exps\.weight', '--out', str(self.out)],
                       check=True, capture_output=True)
        info = mix.parse_gguf(self.out)
        data = self.out.read_bytes()
        want = {name: (gtype, payload) for name, _dims, gtype, payload in self.base_t}
        for name, _dims, gtype, payload in self.donor_t:
            if name == 'blk.0.ffn_down_exps.weight':
                want[name] = (gtype, payload)
        for t in info.tensors:
            gtype, payload = want[t.name]
            self.assertEqual(t.ggml_type, gtype, t.name)
            self.assertEqual(data[t.data_offset:t.data_offset + t.n_bytes], payload, t.name)
        self.assertEqual(info.kv_count, 4)
        for key, value in (('general.mix.base', 'base.gguf'), ('general.mix.donor', 'donor.gguf'),
                           ('general.mix.donor_tensors', r'blk\.0\.ffn_down_exps\.weight')):
            self.assertIn(w_str(key) + struct.pack('<I', 8) + w_str(value), info.kv_blob)


if __name__ == '__main__':
    unittest.main()
