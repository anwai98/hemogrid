# hemogrid

Cell counting for hemocytometer images, with the grid of the chamber as the guide.

Give the program a brightfield tif of a counting chamber. The program finds the grid of the chamber.
It counts the cells in every square. It reports the concentration of the sample. Then it shows the
result in [napari](https://napari.org), so that you can inspect the count.

## Install

```bash
git clone https://github.com/anwai98/hemogrid
cd hemogrid
pip install -e .
```

## Run

```bash
hemogrid chamber.tif
```

The program prints the concentration and the spread of the counts.

```
48,411 cells per microlitre (chamber depth 100 um)
                cells per square
┏━━━━━━━┳━━━━━━━━┳━━━━━┳━━━━━┳━━━━━━━┳━━━━━━━━━┓
┃  mean ┃ median ┃ min ┃ max ┃    cv ┃ squares ┃
┡━━━━━━━╇━━━━━━━━╇━━━━━╇━━━━━╇━━━━━━━╇━━━━━━━━━┩
│ 156.9 │    156 │ 141 │ 172 │ 0.056 │      27 │
└───────┴────────┴─────┴─────┴───────┴─────────┘
6949 cells in the frame: 6816 on amplitude, 133 on the cell profile
```

Then two napari windows open. The first window shows the squares as a stack. It draws a ring on
every cell that the program counted. The second window shows the same cells on the original image.

A run takes about 5 seconds for an image of 1.25 megapixels on 8 processor cores.

## Options

| option | what it does |
| --- | --- |
| `--output-dir results` | Write the tables, the pictures and the segmentation to this directory. |
| `--no-view` | Count the cells, but do not open napari. Use this for a batch of images. |
| `--verbose` | Print what each step found. |
| `--timing` | Print the time that each step took. |
| `--threshold-sigma 2.5` | Lower the threshold to count fainter cells. The default is 3.0. |
| `--depth-um 100` | Set the depth of the chamber. The program needs it for the concentration. |
| `--angle -3.5` | Use this rotation instead of a search. The search takes most of the time. |
| `--workers 1` | Use one thread. The default is one thread for each processor core. |

To see every option, run `hemogrid --help`.
