# pinn-gan-selfheating

A Physics-Informed Neural Network (PINN) framework for GaN HEMT self-heating,
built incrementally from a validated 1D heat-equation skeleton.

**Current milestone:** 1D steady heat equation solved with relative L2 error
`3.23e-05` vs. spec of `1e-2` — a working autograd + PDE-residual + boundary-loss
pipeline in ~3500 parameters, converging in 27 seconds on CPU.

docs/pinn_first_run.png

*Left: PINN prediction (dashed red) overlaid on analytical sin(πx) (solid black).
Right: log-scale loss history showing PDE and BC components.*

---

## What this is

A skeleton PINN implementation designed to be **extended toward realistic GaN
HEMT self-heating simulation**. The 1D steady heat equation is a validated
starting point — chosen because it has a known analytical solution
(`T(x) = sin(πx)`), making it possible to quantify PINN accuracy exactly.

Every design decision in this skeleton — the MLP architecture, the tanh
activation, the boundary-loss weighting, the double-differentiation via
`torch.autograd.grad` — is a stepping stone to the harder physics ahead.

## What this is not (yet)

- Not a production simulator. Classical FEM/FDM still win on speed and reliability
  for problems with well-defined grids.
- Not tuned for real GaN geometry — hotspot resolution, thermal boundary
  resistance, and material stack effects are on the roadmap below.
- Not benchmarked against TCAD. That comes when the physics is closer to
  device-realistic.

---

## Physics roadmap

| # | Extension | New concept | Status |
|---|---|---|---|
| 1 | 1D steady, uniform `k`, analytical BC | Autograd for `d²T/dx²`, soft BC via loss weighting | ✅ Done |
| 2 | 1D transient (`∂T/∂t = α T''`) | Time as an input dimension; initial condition | ⏭️ Next |
| 3 | Temperature-dependent `k(T) = k₀(300/T)^1.4` | Nonlinear PDE; log-transform tricks | ⏭️ |
| 4 | 2D Poisson (`∇²T = Q(x,y)`) | Mesh-free 2D collocation; adaptive sampling (RAR) | ⏭️ |
| 5 | Realistic Joule heating `Q(x,y)` | Coupling to device electrostatics | ⏭️ |
| 6 | Full HEMT stack (substrate + buffer + channel) | Material interfaces; thermal boundary resistance | ⏭️ |

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/siddhanta-roy/pinn-gan-selfheating.git
cd pinn-gan-selfheating

# 2. Environment (Python 3.12, PyTorch 2.12 CPU)
python -m venv .venv
source .venv/bin/activate     # bash/zsh
pip install -r requirements.txt

# 3. Run the baseline (~30 s on CPU)
python -m src.train
```

Expected console output ends with:

```
Training done in ~27 s
Relative L2 error vs analytical: 3.230e-05
Plot saved to: outputs/pinn_first_run.png
```

For interactive exploration:

```bash
jupyter lab notebooks/01_first_experiment.ipynb
```

---

## Project layout

```
pinn-gan-selfheating/
├── src/
│   ├── __init__.py
│   ├── model.py          MLP with Xavier init, tanh activations
│   ├── pde_loss.py       PDE residual + boundary loss via autograd
│   └── train.py          Adam training loop + diagnostics
├── notebooks/
│   └── 01_first_experiment.ipynb   Thin driver + residual diagnostic
├── docs/
│   └── pinn_first_run.png          Milestone 1 result plot
├── outputs/              (gitignored — generated at runtime)
├── models/               (gitignored — checkpoint storage)
├── data/                 (empty — reserved for future datasets)
├── requirements.txt
├── restart.md            Environment restart guide
└── README.md
```

---

## Design notes

A few decisions worth flagging for anyone extending this:

- **Tanh activation.** PDE residual computation differentiates the network twice
  via autograd. ReLU has a zero second derivative, structurally unable to
  represent PDE curvature. Tanh gives smooth `C^∞` gradients through all orders.

- **Xavier initialization.** Keeps activation variance stable across layers at
  init, so Adam has a well-scaled starting point. Bias initialized to zero.

- **Boundary-loss weight `λ = 100`.** Balances the two loss scales, PDE
  residual reaches `O(π² ≈ 10)` early in training while BC residuals are
  `O(0.01)`. Without weighting, Adam ignores the BC. This is the single biggest
  hyperparameter footgun in PINNs.

- **`create_graph=True` in autograd.** Required because we differentiate twice
  (for `T''`) and then backprop through the loss into `θ`. Two-level
  differentiation across different variable types (`x` and `θ`), autograd
  handles it transparently, but only if the derivative graph is preserved.

- **Fixed seed = 42.** Runs are bit-for-bit reproducible. Change with intent.

---

## References

- Raissi, Perdikaris, Karniadakis (2019). *Physics-informed neural networks: A
  deep learning framework for solving forward and inverse problems involving
  nonlinear partial differential equations.* J. Comp. Phys.
- Wang, Yu, Perdikaris (2022). *When and why PINNs fail to train: A neural
  tangent kernel perspective.* J. Comp. Phys.
- Lu, Meng, Mao, Karniadakis (2021). *DeepXDE: A deep learning library for
  solving differential equations.* SIAM Review.

---

## Author

Siddhanta Roy — MTS, Device Modeling & Characterization, GlobalFoundries Bangalore.
Building at the intersection of power semiconductor physics and machine learning.

*This project is part of a broader learning arc: see the companion
[remote-dev-setup-playbook](https://github.com/siddhanta-roy/remote-dev-setup-playbook guide.*