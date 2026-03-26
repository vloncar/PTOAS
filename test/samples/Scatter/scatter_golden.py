#!/usr/bin/python3
import numpy as np
from pathlib import Path
import sys

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / 'validation_runtime.py').is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import default_buffers, float_values, load_case_meta, rng, single_output, write_buffers, write_golden


def main():
    meta = load_case_meta()
    src_name, idx_name = meta.inputs
    generator = rng()
    src_dtype = meta.np_types[src_name]
    src = float_values(generator, meta.elem_counts[src_name], style='signed').astype(src_dtype, copy=False)
    rows = 32
    cols = 64
    row_dest = (np.arange(rows, dtype=np.int64) * 17) % rows
    idx = np.zeros((rows, cols), dtype=np.int64)
    for row in range(rows):
        idx[row, :] = row_dest[row] * cols + np.arange(cols, dtype=np.int64)
    out = np.zeros((rows, cols), dtype=src_dtype)
    out[row_dest, :] = src.reshape(rows, cols)
    buffers = default_buffers(meta)
    buffers[src_name] = src
    buffers[idx_name] = idx.astype(meta.np_types[idx_name], copy=False).reshape(-1)
    write_buffers(meta, buffers)
    write_golden(meta, {single_output(meta): out.reshape(-1)})


if __name__ == '__main__':
    main()
