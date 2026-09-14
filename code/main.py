import argparse
import warnings
from dataclasses import dataclass, field

from utils import Console
from runner import Runner
from displayer import Displayer


# -----------------------------------------------------------------------------
# Datasets Summary
# -----------------------------------------------------------------------------

# --- Classification Summary ---
# tiny_classification = {15: 'Breast Cancer (WI)', 335: 'Monks 3', 42192: 'Compass'}
# low_classification = {13: 'Breast Cancer', 37: 'Pima Indians Diabetes', 44: 'Spambase', 50: 'Tic-Tac-Toe', 56: 'Vote', 1480: 'Indian Liver Patients', 40683: 'Postoperative', 43672: 'Heart Disease', 45578: 'California Housing'}
# med_classification = {24: 'Mushroom', 27: 'Colic', 29: 'Credit Approval', 31: 'Credit-G', 55: 'Hepatitis', 59: 'Ionosphere', 179: 'Adult', 1461: 'Bank Marketing', 23512: 'Higgs', 40981: 'Australian'}
# high_classification = {40: 'Sonar', 1116: 'Musk', 1486: 'Nomao', 40536: 'Speed Dating'}

# --- Regression Summary ---
# tiny_regression = {8: 'Liver Disorders', 230: 'Machine CPU', 42372: 'Auto MPG', 44032: 'Fifa', 44133: 'Pol', 44142: 'Bike Sharing Demand', 44145: 'Sulfur', 44146: 'Medical Charges', 44957: 'Airfoil Self Noise', 44958: 'Auction Verification', 44959: 'Concrete Compressive Strength', 44970: 'Fish Toxicity', 44984: 'CPS88 Wages', 44987: 'Socmob', 44994: 'Cars'}
# low_regression = {194: 'Cleveland', 204: 'Cholesterol', 223: 'Stock Prices', 287: 'Wine Quality', 42225: 'Diamonds', 42363: 'Forest Fires', 42726: 'Abalone', 44024: 'California', 44042: 'Black Friday', 44141: 'Brazilian Houses', 44143: 'NYC Taxi Green', 44962: 'Forest Fires', 44963: 'Physicochemical Protein', 44966: 'Solar Flare'}
# med_regression = {9: 'Automobile', 191: 'Wisconsin', 206: 'Triazines', 511: 'Plasma Retinol', 542: 'Pollution', 566: 'Meta', 1089: 'US Crime', 41021: 'Moneyball', 42352: 'Student Performance', 44019: 'House Sales', 44134: 'Elevator', 44137: 'Ailerons', 44139: 'House 16H', 44969: 'Naval Propulsion Plant', 44973: 'Grid Stability', 44983: 'Miami Housing', 44989: 'Kings County', 46283: 'Appliances Energy Prediction', 46328: 'Seoul Bike Sharing', 46337: 'Conso RTE'}
# high_regression = {42724: 'Online News Popularity', 44965: 'Geographical Origin of Music', 44975: 'Wave Energy', 46132: 'NCI 60 Thioguanine', 46134: 'Acute Myeloid Leukemia', 46139: 'Cancer Drug Response Methylation', 46286: 'Communities and Crime'}


# -----------------------------------------------------------------------------
# Base Configurations
# -----------------------------------------------------------------------------
@dataclass
class BaseConfig:
    """Shared parameters across all experiments."""
    seed: int = 48
    k: int = 5
    m: int = 5000
    sigma: float = 1.0
    # Monte Carlo everywhere: the population metrics are estimated the same way
    # on every benchmark, so that a column is comparable across dimensions. The
    # loss is bounded in [0, 1], so the standard error of the estimate is at
    # most 1 / (2 sqrt(eval_samples)) whatever d.
    gen_method: str = "mc"
    eval_samples: int = 100000
    mip_timeout: int = 120
    n_bins: int = 4
    n_runs: int = 1
    use_bias: bool = True
    explainer_names: list = field(default_factory=lambda: ["IHT", "LIME", "MIP"])
    dataset_ids: list = field(default_factory=list)


@dataclass
class ClassificationConfig(BaseConfig):
    classifier_name: str = "Neural Network"


@dataclass
class RegressionConfig(BaseConfig):
    regressor_name: str = "Neural Network"


ALL_EXPLAINERS = ["CVX", "IHT", "LIME", "MAPLE", "MIP"]
SWEEP_EXPLAINER = ["IHT", "LIME", "MIP"]  
CONVERGENCE_EXPLAINER = ["IHT", "MIP"]  

# Series 1 to 4 run on the same benchmarks: reporting sixteen of them in the
# optimization tables and only half of them in the generalization tables would
# leave the reader wondering which ones were dropped and why.
ALL_CLF_DATASETS = [29, 42192, 40683, 43672, 179, 1486, 40536, 1116]
ALL_REG_DATASETS = [44146, 44024, 44962, 46328, 42352, 46132, 46286, 46139]

# Sweeps run on two benchmarks per task, laid out as a 2 x 2 matrix of panels.
# One sits below the dimension at which dense unanchored explanations overtake
# ours and one above, so that the figure covers both regimes.
# SWEEP_CLF_DATASETS = [179, 40536]      # Adult (179), Speed Dating (40536) 
SWEEP_CLF_DATASETS = [40536]      # COMPAS (42192), Adult (179), 
# SWEEP_REG_DATASETS = [42352, 44024]   # Student Performance (42352), California Housing (44024)
SWEEP_REG_DATASETS = [44024]   # Student Performance (42352), California Housing (44024)

# Budgets. Resolution is finest where relevance actually moves -- going from one
# coefficient to two changes everything, going from six to eight changes little
# -- and the ceiling is set at 8 because a linear function of ten weights is no
# longer something a reader can hold in mind.
K_RANGE = [1, 2, 3, 4, 5, 6, 7, 8]

# Concentrations. With p = exp(-sigma) / (1 + exp(-sigma)) as the flip
# probability, this grid spans the regimes uniformly: p = 0.500 (uniform), 0.378,
# 0.269, 0.119, and 0.018 at sigma = 4, where a neighbourhood of d = 149 keeps
# fewer than three flipped coordinates. Nothing is sampled past that: beyond
# sigma = 5 nearly every draw equals the reference instance, every anchored
# explanation is near perfect, and the regime measures nothing.
SIGMA_RANGE = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

# Sample sizes, spaced by sqrt(10) so that the points are equidistant on the
# log-log plot and the m^{-1/2} slope of the sample complexity bound is legible.
M_RANGE = [100, 500, 1000, 5000, 10000, 50000, 100000]


# -----------------------------------------------------------------------------
# Experiment-Specific Configurations
# -----------------------------------------------------------------------------

# Series 1-2. All sixteen benchmarks, all five explainers, at the common
# operating point k = 5, m = 5000, sigma = 1.
@dataclass
class OptimizationClassificationConfig(ClassificationConfig):
    dataset_ids: list = field(default_factory=lambda: list(ALL_CLF_DATASETS))
    explainer_names: list = field(default_factory=lambda: list(ALL_EXPLAINERS))


@dataclass
class OptimizationRegressionConfig(RegressionConfig):
    dataset_ids: list = field(default_factory=lambda: list(ALL_REG_DATASETS))
    explainer_names: list = field(default_factory=lambda: list(ALL_EXPLAINERS))


# Series 3-4. The decisive series: fidelity buys nothing without anchoring, and
# relevance is where that shows. Same sixteen benchmarks as series 1-2, so that
# the two halves of the section cover the same ground.
@dataclass
class GeneralizationClassificationConfig(ClassificationConfig):
    dataset_ids: list = field(default_factory=lambda: list(ALL_CLF_DATASETS))
    explainer_names: list = field(default_factory=lambda: list(ALL_EXPLAINERS))


@dataclass
class GeneralizationRegressionConfig(RegressionConfig):
    dataset_ids: list = field(default_factory=lambda: list(ALL_REG_DATASETS))
    explainer_names: list = field(default_factory=lambda: list(ALL_EXPLAINERS))


# Series 5-6, two benchmarks per task. Relevance against the budget: empirical
# fidelity decreases with k
# by construction, so only relevance arbitrates a trade-off. MAPLE is absent
# because it takes no budget: its curve would be a horizontal line, and the
# level is already measured at the common operating point in series 1-2, from
# which the figure draws it as a reference.
@dataclass
class KSweepClassificationConfig(ClassificationConfig):
    dataset_ids: list = field(default_factory=lambda: list(SWEEP_CLF_DATASETS))
    k_range: list = field(default_factory=lambda: list(K_RANGE))
    explainer_names: list = field(default_factory=lambda: ["IHT", "MIP"])


@dataclass
class KSweepRegressionConfig(RegressionConfig):
    dataset_ids: list = field(default_factory=lambda: list(SWEEP_REG_DATASETS))
    k_range: list = field(default_factory=lambda: list(K_RANGE))
    explainer_names: list = field(default_factory=lambda: ["IHT", "LIME", "MIP"])


# Series 7-8. Fidelity against relevance, one point per (explainer, sigma), with
# the bound R <= (1 + exp(-sigma))^k F drawn as a line. MAPLE belongs here,
# unlike in the k-sweep: its SILO weights depend on the local distribution, so
# it genuinely varies along the axis.
@dataclass
class SigmaSweepClassificationConfig(ClassificationConfig):
    dataset_ids: list = field(default_factory=lambda: list(SWEEP_CLF_DATASETS))
    sigma_range: list = field(default_factory=lambda: list(SIGMA_RANGE))
    explainer_names: list = field(default_factory=lambda: ["IHT", "LIME", "MIP"])


@dataclass
class SigmaSweepRegressionConfig(RegressionConfig):
    dataset_ids: list = field(default_factory=lambda: list(SWEEP_REG_DATASETS))
    sigma_range: list = field(default_factory=lambda: list(SIGMA_RANGE))
    explainer_names: list = field(default_factory=lambda: ["IHT", "LIME", "MIP"])


# Series 9-10. The generalization gap |F - F_hat| against m, which is what the
# sample complexity theorem bounds. Small benchmarks only: the cost of the MIP
# grows with m, and at m = 100000 the model construction alone becomes
# noticeable.
@dataclass
class MSweepClassificationConfig(ClassificationConfig):
    dataset_ids: list = field(default_factory=lambda: list(SWEEP_CLF_DATASETS))
    m_range: list = field(default_factory=lambda: list(M_RANGE))
    explainer_names: list = field(default_factory=lambda: ["IHT", "LIME", "MIP"])


@dataclass
class MSweepRegressionConfig(RegressionConfig):
    dataset_ids: list = field(default_factory=lambda: list(SWEEP_REG_DATASETS))
    m_range: list = field(default_factory=lambda: list(M_RANGE))
    explainer_names: list = field(default_factory=lambda: ["IHT", "LIME", "MIP"])


# Series 11-12. Anytime behaviour of the exact solver alone, on the two worst
# cases of the campaign, where it never proves optimality within the limit. Run
# last, and possibly for the appendix rather than the main text.
@dataclass
class TimeSweepClassificationConfig(ClassificationConfig):
    dataset_ids: list = field(default_factory=lambda: [1116])
    explainer_names: list = field(default_factory=lambda: ["MIP"])
    min_time: int = 10
    max_time: int = 300
    step_time: int = 10
    n_runs: int = 5


@dataclass
class TimeSweepRegressionConfig(RegressionConfig):
    dataset_ids: list = field(default_factory=lambda: [46139])
    explainer_names: list = field(default_factory=lambda: ["MIP"])
    min_time: int = 10
    max_time: int = 300
    step_time: int = 10
    n_runs: int = 5


# Series 13-14. The ratio of empirical fidelities against sigma, which is the
# form the end-to-end theorem takes. Benchmarks small enough for the MIP to
# prove optimality, so that the denominator is the true optimum -- the four
# dimension levels used elsewhere would make the ratio uninterpretable.
@dataclass
class ConvergenceClassificationConfig(ClassificationConfig):
    dataset_ids: list = field(default_factory=lambda: [29, 42192])
    explainer_names: list = field(default_factory=lambda: ["IHT", "MIP"])
    sigma_range: list = field(default_factory=lambda: list(SIGMA_RANGE))


@dataclass
class ConvergenceRegressionConfig(RegressionConfig):
    dataset_ids: list = field(default_factory=lambda: [44146, 44024])
    explainer_names: list = field(default_factory=lambda: ["IHT", "MIP"])
    sigma_range: list = field(default_factory=lambda: list(SIGMA_RANGE))


# -----------------------------------------------------------------------------
# Series registry
# -----------------------------------------------------------------------------

SERIES = {
    1: ("Optimization for Classification", OptimizationClassificationConfig, "optimization_clf"),
    2: ("Optimization for Regression", OptimizationRegressionConfig, "optimization_reg"),
    3: ("Generalization for Classification", GeneralizationClassificationConfig, "generalization_clf"),
    4: ("Generalization for Regression", GeneralizationRegressionConfig, "generalization_reg"),
    5: ("K-Sweep for Classification", KSweepClassificationConfig, "k_sweep_clf"),
    6: ("K-Sweep for Regression", KSweepRegressionConfig, "k_sweep_reg"),
    7: ("Sigma-Sweep for Classification", SigmaSweepClassificationConfig, "sigma_sweep_clf"),
    8: ("Sigma-Sweep for Regression", SigmaSweepRegressionConfig, "sigma_sweep_reg"),
    9: ("M-Sweep for Classification", MSweepClassificationConfig, "m_sweep_clf"),
    10: ("M-Sweep for Regression", MSweepRegressionConfig, "m_sweep_reg"),
    11: ("Time-Sweep for Classification", TimeSweepClassificationConfig, "time_sweep_clf"),
    12: ("Time-Sweep for Regression", TimeSweepRegressionConfig, "time_sweep_reg"),
    13: ("Convergence for Classification", ConvergenceClassificationConfig, "convergence_clf"),
    14: ("Convergence for Regression", ConvergenceRegressionConfig, "convergence_reg"),
}

# Each family maps to the runner method that produces the CSV and the displayer
# method that turns it into the table or figure of the paper.
HANDLERS = {
    "optimization": ("run_optimization_experiment", "generate_optimization_table"),
    "generalization": ("run_generalization_experiment", "generate_generalization_table"),
    "k_sweep": ("run_k_sweep_experiment", "generate_k_sweep_plot"),
    "sigma_sweep": ("run_sigma_sweep_experiment", "generate_sigma_sweep_plot"),
    "m_sweep": ("run_m_sweep_experiment", "generate_m_sweep_plot"),
    "time_sweep": ("run_time_sweep_experiment", "generate_time_sweep_plot"),
    "convergence": ("run_convergence_experiment", "generate_convergence_plot"),
}


def _family(task_type: str) -> str:
    for family in HANDLERS:
        if task_type.startswith(family):
            return family
    raise ValueError(f"No handler for {task_type}")


def _resolve(tokens: list) -> list:
    """Accepts series numbers, series names, a family name, or 'all'."""
    if not tokens:
        return []
    if any(str(token).lower() == "all" for token in tokens):
        return sorted(SERIES)

    by_name = {task_type: key for key, (_, _, task_type) in SERIES.items()}
    selected = []
    for token in tokens:
        token = str(token).strip().strip(",")
        if not token:
            continue
        if token.isdigit() and int(token) in SERIES:
            selected.append(int(token))
        elif token in by_name:
            selected.append(by_name[token])
        else:
            # A family name selects both of its tasks, e.g. "k_sweep".
            matches = [key for key, (_, _, t) in SERIES.items() if t.startswith(token)]
            selected.extend(matches if matches else [token])
    return selected


def run_series(keys: list, console: Console) -> None:
    runner = Runner(console=console)
    displayer = Displayer(output_dir="./exps")

    for key in keys:
        if key not in SERIES:
            console.log(f"Warning: [{key}] is not a valid series and will be skipped.", color="yellow")
            continue

        name, ConfigClass, task_type = SERIES[key]
        console.log(f"Starting: {name}", color="green")

        config = ConfigClass()
        is_clf = task_type.endswith("_clf")
        run_method, display_method = HANDLERS[_family(task_type)]

        output_path, _ = getattr(runner, run_method)(config=config, is_classification=is_clf)
        getattr(displayer, display_method)(str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the experimental series of the paper.")
    parser.add_argument(
        "--series",
        nargs="*",
        default=None,
        help="Series numbers, names (e.g. k_sweep_clf), a family (e.g. k_sweep), or 'all'. "
             "Omit to get the interactive menu.",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    console = Console(verbose=True)

    if args.series is not None:
        keys = _resolve(args.series)
    else:
        console.log("Benchmark Menu", color="green")
        for key, (name, _, _) in SERIES.items():
            print(f"[{key}] {name}")
        print("[0] Exit")
        print("-" * 36)
        keys = _resolve(input("Enter the labels of the series you want to run (e.g., 1, 3): ").replace(",", " ").split())
        if 0 in keys:
            keys = []

    if not keys:
        console.log("Exiting.", color="green")
        return

    run_series(keys, console)
    console.log("All Experiments Completed", color="green")


if __name__ == "__main__":
    main()