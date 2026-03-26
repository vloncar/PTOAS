#!/usr/bin/python3
import numpy as np
from pathlib import Path
import sys

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / 'validation_runtime.py').is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import (
    default_buffers,
    float_values,
    is_a5_soc,
    load_case_meta,
    rng,
    single_output,
    write_buffers,
    write_golden,
)


def main():
    meta = load_case_meta()
    src_name, offset_name = meta.inputs
    generator = rng()
    rows = 32
    cols = 32
    src_dtype = meta.np_types[src_name]
    row_values = float_values(generator, rows, style='signed_small')
    src = np.repeat(row_values[:, None], cols, axis=1).astype(src_dtype, copy=False).reshape(-1)
    offsets = np.zeros((rows, cols), dtype=meta.np_types[offset_name])
    for row in range(rows):
        offsets[row, :] = row * cols * src_dtype.itemsize
    if is_a5_soc():
        expected = src
    else:
        block_elems = 32 // src_dtype.itemsize
        row_repeat = cols // block_elems
        expected_rows = row_values[np.arange(rows) // row_repeat]
        expected = np.repeat(expected_rows[:, None], cols, axis=1).astype(src_dtype, copy=False).reshape(-1)
    buffers = default_buffers(meta)
    buffers[src_name] = src
    buffers[offset_name] = offsets.reshape(-1)
    write_buffers(meta, buffers)
    write_golden(meta, {single_output(meta): np.asarray(expected, dtype=src_dtype)})


if __name__ == '__main__':
    main()
