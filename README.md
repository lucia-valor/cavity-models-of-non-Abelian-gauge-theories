# Cavity models of non-Abelian gauge theories — numerical figures

Code for the numerical figures of *Cavity models of non-Abelian gauge theories*,
L. Valor, P. Luneau, K. Liegener, S. Filipp, P. Rabl.

`figures.ipynb` produces Figs. 3–6; Figs. 1, 2, 7 and 8 are drawings. Run it top to
bottom: it takes under a minute, using the Fig. 6 ground states stored in `data/`.

| File | Contents |
|---|---|
| `figures.ipynb` | parameters, data and plots for every figure |
| `driven_cavity.py` | Figs. 3–4: single driven cavity in the $(N, l, m=0)$ basis; effective glueball model |
| `bound_states.py` | Fig. 5: two-glueball problem in the relative coordinate |
| `glueball_chain.py` | Fig. 6: many glueballs on a ring of cavities |
| `plotting.py` | figure styles and layout |
| `data/` | Fig. 6 ground states, `fig6_ground_state_ng{2,3,4,5}.npz` |

The notebook writes `data/` itself: set `regenerate = True` in the Fig. 6 data cell to
recompute the ground states (about 6 min, dominated by the dense diagonalisation for
$\mathcal{N} = 5$). Each file also records the parameters it was made with.

## Requirements

Python ≥ 3.11, the packages in `requirements.txt`, and a LaTeX installation for the
figure text.

```bash
pip install -r requirements.txt
jupyter lab figures.ipynb
```
