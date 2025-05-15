
# Hamiltonian Library Generator & Simulated Annealing Solver

This repository contains two Python scripts:

| File | Purpose |
|------|---------|
| **`libgen.py`** | Generates sparse Hamiltonian matrices for Ising‑style optimization problems. |
| **`annealing.py`** | Runs **simulated annealing** on the matrices created by `libgen.py`. |

**Execution order:** **`libgen.py` → `annealing.py`**

---

## File Structure

```
.
├─ libgen.py
├─ annealing.py
└─ matrix/           # Created automatically by libgen.py
```

---

## Requirements

- Python ≥ 3.8  
- NumPy ≥ 1.23  
- SciPy ≥ 1.10  
- Pandas ≥ 2.0  
- Matplotlib ≥ 3.8  
- tqdm (optional, for progress bars)

```bash
# Recommended: create a virtual environment first
python -m venv venv
source venv/bin/activate        # On Windows: .env\Scriptsctivate
pip install numpy scipy pandas matplotlib tqdm
```

---

## Quick Start

```bash
# 1. Generate Hamiltonian matrices
python libgen.py

# 2. Run simulated annealing with the default "normal" preset
python annealing.py

# Optional: use the "quick" preset for faster but less thorough runs
python annealing.py --preset quick
```

`libgen.py` outputs `.npz`, `.csv`, and `.feature.csv` files under `matrix/<graph_type>/`.  
`annealing.py` automatically reads the latest files in that folder.

---

## Configuring `libgen.py`

Inside `libgen.py` you’ll find lists at the top of `main()` that control batch generation:

| Variable | Description | Example |
|----------|-------------|---------|
| `type_list` | Gate sets to combine (e.g., `HNN`, `HAA`) | `["HNN"]` |
| `size_list` | Problem sizes (number of spins) | `[100, 250, 500]` |
| `scale_list` | Weight scaling factors | `[0.5, 1.0, 2.0]` |
| `scale_mode_list` | Scaling mode (`normal`, `uniform`, `fixed`) | `["fixed"]` |
| `seed` | Random seed(s) | `[42]` |

Adjust the lists and rerun the script to create multiple datasets in one go.

---

## Command‑Line Options for `annealing.py`

```
--preset {quick, normal}   Select a predefined parameter set (default: normal)
--trial INT                Override the number of trials
--cycle INT                Override cycles per trial
```

You can add or edit presets in the `_PRESETS` dictionary (around line 400).

---

## Output

* Energy histories and spin‑state statistics are printed to the console.  
* If `verbose_plot="yes"` is enabled, PNG plots of the energy trajectory are saved automatically.

---

## License

This project is released under the MIT License.  
Please add a `LICENSE` file before publishing the repository.
