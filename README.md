# collective-portfolio-lab

Independent toolkit and CLI inspired by Azar Aliyev, "Theory of Household Portfolio Choice: Pitfalls in Applications of the Collective Model", arXiv:2608.12411v1.

**Not official author code. Not a claim of exact bit-for-bit replication.**

This repository independently implements the binary gamble and continuous-choice collective-model examples described in the paper for research reproducibility. It is a deterministic, alpha-in-[0,1], Figure-1-style independent variant — not bit-for-bit author code.

Canonical paper URL: https://arxiv.org/abs/2608.12411v1

## What this is (independent-implementation disclaimer)

- **Continuous model:** Normal-excess-return simulation (`x ~ N(mu, sigma^2)`), bounded risky share `alpha` in `[0, 1]`, deterministic common random numbers (`seed=42`, `n_sims=20000` by default).
- **Positivity handling:** CRRA simulation validates strictly positive consumption; if any draw yields non-positive consumption, the objective returns a very negative value (`-1e12`) rather than silently evaluating at invalid inputs.
- **Aggregation:** Additive household aggregation (`lambda1 * E[U1] + lambda2 * E[U2]`) is one alternative; the analytic CARA expected-product benchmark (`alpha* = mu / (sum lambda_i A_i W sigma^2)`) is provided for comparison. The benchmark is reported as the raw analytic value; clipping is documented explicitly and never applied silently.
- **Non-monotonicity:** Tolerance-based detector (`tolerance=1e-4`) requiring at least one meaningful decrease followed later by a meaningful increase; it does not require adjacent sign flips and rejects flat numerical noise.
- **Gamma=1 convention:** For CRRA utility, `gamma=1` is handled as an explicit `log(C)` convention. It is not claimed continuous with `C^(1-gamma)/(1-gamma)` at `gamma=1`; the singular value is documented as a convention for the paper's utility normalization.
- **Financial advice:** This toolkit demonstrates the model for education and replication; it does **not** provide financial advice.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Quick start (normal API + CLI)

Normal Python API (installed package):

```python
from collective_portfolio_lab import binary_risky_utility, sweep_continuous
import numpy as np

# Binary gamble
binary_risky_utility(gamma1=0.08, gamma2=0.8)

# Continuous sweep (CRRA)
sweep_continuous(
    "crra", param1_fixed=4.0,
    param_range2=np.linspace(0.5, 12.0, 60),
)
```

CLI commands:

```bash
python -m collective_portfolio_lab binary-example --format json
python -m collective_portfolio_lab reproduce --output-dir ./results
python -m collective_portfolio_lab scan --utility cara --lower 0.5 --upper 8.0 --points 60 --format csv --output scan.csv
collective-portfolio binary-example --format json
```

Every snippet and command above has been executed in this repository after editing.

## Use cases (executed)

### 1. Identification ambiguity — `binary-example`

Exact risky/safe/risky result:

```bash
python -m collective_portfolio_lab binary-example --format json
```

Observed output (actual run):

```json
[{"gamma2":0.08,"risky_expected_utility":11.356813185697742,"safe_utility":9.556405560227063,"difference":1.8004076254706796,"choice":"risky"},{"gamma2":0.8,"risky_expected_utility":11.3123731704819,"safe_utility":11.676851087419607,"difference":-0.36447791693770704,"choice":"safe"},{"gamma2":8.0,"risky_expected_utility":5.549835164277438,"safe_utility":4.778200951542103,"difference":0.7716342127353348,"choice":"risky"}]
```

The binary choice sequence is `risky -> safe -> risky`, confirming the identification-ambiguity reversal described in the paper.

### 2. Replication workflow — `reproduce`

```bash
python -m collective_portfolio_lab reproduce --output-dir ./results
```

This generates exactly six artifacts in `./results/`:

- `binary_example.csv`
- `figure1_cara.csv`
- `figure1_crra.csv`
- `figure1_cara.png`
- `figure1_crra.png`
- `summary.json`

Reading `summary.json` (actual file):

```bash
cat ./results/summary.json
```

Output:

```json
{
  "model_parameters": {
    "mu": 0.1,
    "sigma": 0.2,
    "rf": 0.0,
    "wealth": 2.0,
    "lambda1": 0.5,
    "lambda2": 0.5,
    "simulation_count": 20000,
    "seed": 42,
    "points": 60
  },
  "non_monotonicity": {
    "cara_detected": true,
    "crra_detected": true
  },
  "benchmark": {
    "expected_product_cara_midpoint": 0.3999999999999999
  },
  "notes": "Independent implementation inspired by arXiv:2608.12411v1. Not official author code."
}
```

### 3. Robustness comparison — importable Python API

Compare additive household choice with the equation-13 CARA benchmark using the public API:

```python
from collective_portfolio_lab import sweep_continuous, cara_expected_product_benchmark
import numpy as np

res = sweep_continuous(
    "cara", param1_fixed=2.0,
    param_range2=np.linspace(0.5, 8.0, 60),
    lambda1=0.5, lambda2=0.5,
)
benchmark = cara_expected_product_benchmark(
    A1=2.0, A2=float(np.median(np.linspace(0.5, 8.0, 60))),
    lambda1=0.5, lambda2=0.5,
    mu=0.1, sigma=0.2, W=2.0,
)
print("Household share at midpoint:", res["household_share"][29])
print("Benchmark:", benchmark)
```

Actual run output:

```
Household share at midpoint: 0.6153714356026843
Benchmark: 0.3999999999999999
```

The benchmark (`0.4`) is the raw analytic unconstrained value from equation (13); the additive household share (`~0.615`) is higher, showing the divergence between the two aggregation assumptions.

### 4. Teaching / custom sweep — `scan`

```bash
python -m collective_portfolio_lab scan --utility cara --lower 0.5 --upper 8.0 --points 5 --format json
```

Actual output excerpt:

```json
[
  {"risk_aversion_2": 0.5, "individual_1_share": 0.6249999999999999, "individual_2_share": 1.0, "household_share": 0.9999999700466811},
  {"risk_aversion_2": 2.375, "individual_1_share": 0.6249999999999999, "individual_2_share": 0.5263157894736842, "household_share": 0.5895052124136653},
  ...
]
```

Note that `individual_2_share` at `risk_aversion_2=0.5` is clipped to `1.0` because the raw analytic value (`2.5`) exceeds the bounded interval `[0, 1]`. The clipping is explicit and documented; the equation itself (`alpha* = mu / (A*W*sigma^2)`) is unchanged.

## Normal Python API quick-start

```python
from collective_portfolio_lab import binary_risky_utility, sweep_continuous
import numpy as np

# Binary gamble (exact three-point result)
results = [binary_risky_utility(0.08, g) for g in [0.08, 0.8, 8.0]]

# Continuous sweep with fixed seed and documented simulation count
sweep_continuous(
    "crra", param1_fixed=4.0,
    param_range2=np.linspace(2.0, 8.0, 10),
    lambda1=0.5, lambda2=0.5,
    seed=42, n_sims=20000,
)
```

## Assumptions and limitations

- Normal-return assumption (`x ~ N(mu, sigma^2)`).
- Additive household aggregation (`sum lambda_i E[U_i]`) is one alternative; the expected-product benchmark (`alpha* = mu / (sum lambda_i A_i W sigma^2)`) is provided for comparison.
- CRRA simulation uses common random numbers with an explicit fixed seed (`seed=42`) and documented simulation count (`n_sims=20000` by default). It validates positive consumption explicitly; if non-positive consumption occurs, the objective returns `-1e12` rather than evaluating at invalid inputs.
- Non-monotonicity detection uses a numerical tolerance (`1e-4`) rather than exact floating-point equality; it requires a meaningful decrease followed later by a meaningful increase.
- This toolkit demonstrates the model for education and replication; it does **not** provide financial advice.
- The continuous implementation is deterministic, bounded (`alpha` in `[0,1]`), and independently constructed — not bit-for-bit author code.

## Testing

```bash
.venv/bin/pytest -q
```

Tests cover binary values, validation (`n_sims` positive integer, positive finite risk aversion, non-finite inputs), scalar optimization agreement with dense-grid reference, non-monotonicity tolerance controls, and CLI smoke tests.

## License

MIT. See `LICENSE` and `CITATION.cff`.
