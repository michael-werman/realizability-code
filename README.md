# Which Images Can a Family of 3D Shapes Produce? — code

Code accompanying

> M. Werman, *Which Images Can a Family of 3D Shapes Produce? Realizability, Boundaries and Focal Length under Perspective*, 2026.

Every script is self-contained and reproduces one part of the paper.

| Script | What it reproduces | Runtime |
| --- | --- | --- |
| `verify_identities.py` | Symbolic and numerical checks of the identities behind the main results (Lemma on tangent cones, the ellipsoid theorem, the interval proposition, the corner and sphere formulas, the certified test, the comparison with Gaudillière et al.) | about a minute |
| `make_figures.py` | The figures of the paper except the composite (written to `figures/`; `fig4_ellipsoid.pdf` is no longer used in the paper) and the values of the worked ellipse example | seconds |
| `certified.py` | Section 9, certified ellipse test (table of certainly consistent / certainly rejected rates) | a few minutes |
| `pshift.py` | Section 9, sensitivity to an error in the principal point (needs `certified.py` in the same folder) | a few minutes |
| `composite.py` | The composite-image example and Figure 8 | seconds |
| `rank.py` | Supplementary table: generic rank of the image map for corners, spheres, ellipsoids and rectangles | seconds |

## Running

```bash
python -m pip install -r requirements.txt
python verify_identities.py
python make_figures.py
python certified.py
python pshift.py
python composite.py
python rank.py
```

Random seeds are fixed, so the numbers match those in the paper.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).

## License

MIT, see `LICENSE`.
