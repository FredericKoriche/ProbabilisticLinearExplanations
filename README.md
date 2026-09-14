# Global Probabilistic Explanations and Sparse Optimization for AI

This repository contains the official Python implementation and experimental pipeline for our paper submitted to the **Artificial Intelligence Journal (AIJ)**. 

The codebase provides exact and approximate algorithms for sparse linear model explanation under anchoring constraints, including Mixed-Integer Programming (MIP), its convex relaxation (CVX), and Iterative Hard Thresholding (IHT), alongside the baselines LIME and MAPLE.

---

## Repository Structure

The project is structured within a single `code` directory containing all core modules and entry points:

    code/
    ├── blackbox.py   # Black-box classifiers and regressors (PyTorch/Scikit-Learn bridges)
    ├── dataset.py    # OpenML dataset downloading, cleaning, and binarization pipeline
    ├── displayer.py  # Automated LaTeX table and PDF/PNG figure generation
    ├── explainer.py  # Attribution explainers: MIP, IHT, CVX, LIME, and MAPLE
    ├── main.py       # Command-line interface for running the experimental series
    ├── runner.py     # Orchestration layer for execution loops and metric collection
    ├── sampler.py    # L1-norm parameterized PyTorch neighborhood sampler
    └── utils.py      # Console logging and hardware/architecture utilities

---

## Requirements & Installation

The code requires Python 3.14+ and relies on **PyTorch**, **Gurobi** (required for MIP and CVX optimization), **Scikit-Learn**, and standard data science libraries. We recommend using **Mamba** for fast environment creation and package management.

*Note: Generating the PDF figures requires a local LaTeX distribution (e.g., TeX Live, MiKTeX, or MacTeX) installed on your system with the `amsmath`, `amsfonts`, `amssymb`, and `bm` packages.*

1. **Clone the repository and navigate to the code directory:**
       cd code

2. **Create and activate a Mamba environment:**
       mamba create -n aij-exps python=3.14
       mamba activate aij-exps

3. **Install dependencies:**
       mamba install pytorch scikit-learn pandas numpy matplotlib seaborn cvxpy lime colorama psutil -c conda-forge
       mamba install gurobi -c gurobi

   *Note: To enable GPU acceleration, ensure you install the PyTorch version compiled with the appropriate CUDA toolkit for your hardware.*
   
   *Note: Gurobi requires an active license (free academic licenses are available).*

---

## Running Experiments

All experimental series described in the paper can be executed via `main.py`. 

### Interactive Menu
To launch an interactive menu where you can choose which experimental series to run:
    python main.py

### Command-Line Execution
You can also run specific series directly using their identifiers or family names:
* **Optimization tables (Classification):**
      python main.py --series 1

* **Generalization evaluation (Regression):**
      python main.py --series 4

* **Sparsity ($k$) sweep:**
      python main.py --series k_sweep

* **Run all experiments:**
      python main.py --series all

Outputs (raw CSV data, generated LaTeX tables, and PDF/PNG plots) are automatically saved to the `./exps/` directory.

---

## License

This project is licensed under the terms of the **MIT License**.