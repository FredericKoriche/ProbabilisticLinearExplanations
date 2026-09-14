# -----------------------------------------------------------------------------
#
# Runner Class
#
# Organised in four layers: outputs, pipeline construction, metric collection,
# and the experiment drivers. The two drivers -- the generic sweep and the time
# sweep -- share _prepare_run and _metric_row, so that a column added once shows
# up in every series.
#
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import time
import torch

from pathlib import Path
from typing import List, Dict, Any, Tuple

from blackbox import Classifier, Regressor
from dataset import ClassificationDataset, RegressionDataset
from explainer import CVXExplainer, IHTExplainer, LIMEExplainer, MAPLEExplainer, MIPExplainer
from sampler import Sampler
from utils import Console


class Runner:
    # Which inputs each explainer expects. LIME needs the model itself, since it
    # resamples on its own; MAPLE takes no anchoring point.
    EXPLAINER_INPUTS = {
        "CVX": ("x", "y", "Z", "f_Z", "maxsize", "features"),
        "IHT": ("x", "y", "Z", "f_Z", "maxsize", "features"),
        "MIP": ("x", "y", "Z", "f_Z", "maxsize", "features"),
        "LIME": ("x", "Z", "maxsize", "features", "model"),
        "MAPLE": ("x", "Z", "f_Z", "maxsize", "features"),
    }

    # Metrics only some explainers expose, as {csv column: explainer attribute}.
    # Absent attributes are recorded as None rather than omitted, so that every
    # row of the CSV carries the same columns.
    OPTIONAL_METRICS = {
        # IHT: how far the iteration cap is from being reached, and whether the
        # step size ever let the empirical fidelity increase.
        "n_iterations": "n_iterations",
        "monotone": "monotone",
        # MIP: is_optimal is what licenses the word "exact" -- without it the
        # column is an incumbent like any other.
        "is_optimal": "is_optimal",
        "mip_gap": "mip_gap",
        # Objective values in F_hat units, straight from the solvers. For CVX
        # this is a certified lower bound on the optimum of the MIP, the only
        # guarantee available where the MIP hits its time limit.
        "solver_fidelity": "fidelity_value",
        "fidelity_bound": "fidelity_bound",
        # MAPLE: the support size its own feature selection settles on.
        "retain": "retain",
        # Magnitude of the anchoring violation, next to its binary verdict.
        "anchoring_gap": "anchoring_gap",
    }

    def __init__(self, console: Console):
        self.console = console
        self.output_base = Path("./exps")
        self.output_base.mkdir(parents=True, exist_ok=True)
        self.sampler = Sampler(console=self.console)

    # -- Outputs ---------------------------------------------------------------

    def _get_output_path(self, series_name: str, prefix: str) -> Path:
        series_dir = self.output_base / f"{series_name}"
        series_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        return series_dir / f"{prefix}_{timestamp}.csv"

    def _save_results(self, results: List[Dict], datasets: list, prefix: str) -> Tuple[Path, str]:
        df = pd.DataFrame(results)
        ds_label = datasets[0].name if len(datasets) == 1 else "multiple_datasets"
        output_path = self._get_output_path(prefix, f"{prefix}_{ds_label}")
        df.to_csv(output_path, index=False)
        self.console.log(f"Saved raw results to {output_path}", color="green")
        return output_path, ds_label

    # -- Pipeline --------------------------------------------------------------

    @staticmethod
    def _settings(config: Any) -> Dict[str, Any]:
        """
        The handful of parameters every driver reads off the config, resolved in
        one place so that a new one does not have to be threaded through twice.
        """
        return {
            # "auto" evaluates the population metrics exactly when the number of
            # free coordinates is small enough, and by Monte Carlo otherwise;
            # any other value forces Monte Carlo. This governs the evaluation of
            # the metrics, not the construction of the explainers.
            "gen_method": getattr(config, "gen_method", "auto"),
            # Monte Carlo budget for the population metrics. The loss is bounded
            # in [0, 1], so the standard error of the estimate is at most
            # 1 / (2 sqrt(eval_samples)) whatever the dimension.
            "eval_samples": getattr(config, "eval_samples", 100_000),
            "mip_timeout": getattr(config, "mip_timeout", 10),
            "seed": getattr(config, "seed", 48),
            "use_bias": getattr(config, "use_bias", True),
        }

    def _prepare_pipeline(self, dataset_ids: list, is_classification: bool, config: Any) -> tuple:
        if isinstance(dataset_ids, int):
            dataset_ids = [dataset_ids]

        seed = getattr(config, "seed", 48)
        datasets = []
        models = []

        self.console.log(
            f"Preparing {'classification' if is_classification else 'regression'} pipeline",
            color="magenta",
        )
        for d_id in dataset_ids:
            if is_classification:
                ds = ClassificationDataset(console=self.console, id=d_id, n_bins=config.n_bins, verbose=False)
                model = Classifier(name=getattr(config, 'classifier_name', 'Neural Network'), seed=seed, verbose=False)
            else:
                ds = RegressionDataset(console=self.console, id=d_id, n_bins=config.n_bins, verbose=False)
                model = Regressor(name=getattr(config, 'regressor_name', 'Neural Network'), seed=seed, verbose=False)

            ds.build()
            model.train(instances=ds.X, labels=ds.Y)
            datasets.append(ds)
            models.append(model)

        return datasets, models

    def _draw_reference_instances(self, ds: Any, n_runs: int, seed: int) -> torch.Tensor:
        """
        Draws the reference instances for one dataset.

        The draw uses a dedicated generator so that it does not depend on how many
        samples the Sampler has already consumed from the global RNG: the same
        instances are then used by every experiment run with the same seed,
        which is what makes the table and the sweeps comparable at their common
        operating point.
        """
        generator = torch.Generator().manual_seed(seed)
        n_instances = ds.X_tensor.shape[0]
        n_drawn = min(n_runs, n_instances)
        if n_drawn < n_runs:
            self.console.log(
                f"{ds.name} holds only {n_instances} instances: "
                f"using {n_drawn} reference instances instead of {n_runs}",
                color="yellow",
            )
        return torch.randperm(n_instances, generator=generator)[:n_drawn]

    def _generate_data(self, x: torch.Tensor, model: Any, sigma: float, m: int) -> tuple:
        Z = self.sampler.sample(x, sigma=sigma, n_samples=m)
        f_Z = model.predict(Z)
        return Z, f_Z

    def _prepare_run(
        self, ds: Any, model: Any, ds_index: int, run: int, idx: int, sigma: float, m: int, seed: int
    ) -> Tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]:
        """
        Everything a single explanation task needs: the reference instance, the
        prediction to anchor on, and the local sample.

        Z depends only on (x, sigma, m), never on k. Reseeding from the dataset
        and the run alone therefore yields the very same sample at every point
        of a k-sweep, so that the resulting curve isolates the effect of the
        budget. For the sigma- and m-sweeps the sample necessarily differs, but
        the underlying draws stay coupled.
        """
        x_test = ds.X_tensor[idx]
        y_test = model.predict(x_test.unsqueeze(0)).item()

        torch.manual_seed(seed + 1000 * ds_index + run)
        Z, f_Z = self._generate_data(x_test, model, sigma, m)
        return x_test, y_test, Z, f_Z

    def _build_explainers(
        self, explainer_names: list, mip_timeout: int, seed: int, sigma: float, use_bias: bool
    ) -> Dict[str, Any]:
        # use_bias is passed only to the explainers that search W: LIME and
        # MAPLE fit whatever intercept their own method prescribes.
        explainers = {}
        if "CVX" in explainer_names:
            explainers["CVX"] = CVXExplainer(use_bias=use_bias, verbose=False)
        if "IHT" in explainer_names:
            explainers["IHT"] = IHTExplainer(sigma=sigma, use_bias=use_bias, verbose=False)
        if "LIME" in explainer_names:
            explainers["LIME"] = LIMEExplainer(seed=seed, verbose=False)
        if "MAPLE" in explainer_names:
            explainers["MAPLE"] = MAPLEExplainer(seed=seed, verbose=False)
        if "MIP" in explainer_names:
            explainers["MIP"] = MIPExplainer(timeout=mip_timeout, use_bias=use_bias, verbose=False)
        return explainers

    # -- Metrics ---------------------------------------------------------------

    def _explainer_kwargs(self, name: str, **available) -> Dict[str, Any]:
        if name not in self.EXPLAINER_INPUTS:
            raise ValueError(f"Unknown explainer: {name}")
        return {key: available[key] for key in self.EXPLAINER_INPUTS[name]}

    def _metric_row(
        self, name: str, explainer: Any, explanation: dict, walltime: float,
        x: torch.Tensor, y: float, Z: torch.Tensor, f_Z: torch.Tensor
    ) -> Dict[str, Any]:
        # The MIP and its relaxation can now come back empty-handed. Their
        # metrics are recorded as NaN rather than scored on a zero vector, which
        # would count as an explanation violating the anchoring.
        solved = bool(getattr(explainer, "solved", True))
        if solved:
            emp_fidelity = explainer.test_empirical_fidelity(Z_test=Z, f_Z_test=f_Z)
            anchoring_violation = explainer.test_anchoring_violation(x=x, y=y)
        else:
            emp_fidelity = float("nan")
            anchoring_violation = float("nan")
            self.console.log(
                f"[{name}] returned no solution: metrics recorded as NaN", color="red"
            )

        intercept = getattr(explainer, "intercept", 0.0)
        intercept = 0.0 if intercept is None else float(intercept)
        sparsity = len(explanation) + (1 if abs(intercept) > 1e-5 else 0)

        # Largest coefficient in absolute value, bias included, for every
        # explainer rather than for IHT alone. The box is a constraint of W, so
        # how far the unconstrained baselines sit outside it is what says
        # whether that constraint binds at all.
        coefficients = getattr(explainer, "solution", None)
        if coefficients is None:
            max_abs_weight = None
        else:
            if hasattr(coefficients, "detach"):
                coefficients = coefficients.detach().cpu().numpy()
            coefficients = np.asarray(coefficients, dtype=float)
            largest = float(np.abs(coefficients).max()) if coefficients.size else 0.0
            max_abs_weight = max(largest, abs(intercept))

        row = {
            "explainer": name,
            "anchoring_violation": anchoring_violation,
            "emp_fidelity": emp_fidelity,
            "intercept": intercept,
            "max_abs_weight": max_abs_weight,
            # Support of the explanation, so that two explainers can be compared
            # on which coordinates they select and not only on how many. The
            # bias is not listed: what is compared are coordinates of z.
            "support": "|".join(sorted(explanation.keys())),
            "solved": solved,
            "sparsity": sparsity,
            "walltime": walltime,
        }
        row.update({
            column: getattr(explainer, attribute, None)
            for column, attribute in self.OPTIONAL_METRICS.items()
        })
        return row

    def _evaluate_optimization_metrics(
        self, x, y, Z, f_Z, maxsize, features, model, explainers: Dict[str, Any], **kwargs
    ) -> List[Dict]:
        results = []
        for name, explainer in explainers.items():
            exp_kwargs = self._explainer_kwargs(
                name, x=x, y=y, Z=Z, f_Z=f_Z, maxsize=maxsize, features=features, model=model
            )
            explanation, walltime = explainer.explain(**exp_kwargs)
            results.append(
                self._metric_row(name, explainer, explanation, walltime, x, y, Z, f_Z)
            )
        return results

    def _evaluate_generalization_metrics(
        self, x, y, Z, f_Z, maxsize, features, model, sigma,
        explainers: Dict[str, Any], gen_method="auto", eval_samples: int = 100_000,
        **kwargs
    ) -> List[Dict]:
        results = self._evaluate_optimization_metrics(
            x, y, Z, f_Z, maxsize, features, model, explainers
        )

        for metric in results:
            explainer = explainers[metric["explainer"]]
            metric["relevance"] = explainer.test_relevance(
                x=x, f_x=y, sigma=sigma, model_predict=model.predict,
                method=gen_method, num_samples=eval_samples,
            )
            metric["fidelity"] = explainer.test_fidelity(
                x=x, sigma=sigma, model_predict=model.predict,
                method=gen_method, num_samples=eval_samples,
            )

        return results

    # -- Experiment drivers ----------------------------------------------------

    def _execute_experiment(
        self, config, is_classification, exp_name, prefix, eval_func,
        sweep_key=None, sweep_values=None
    ):
        self.console.log(f"Running {exp_name}", color="cyan")
        settings = self._settings(config)
        datasets, models = self._prepare_pipeline(config.dataset_ids, is_classification, config)
        all_results = []

        # If no sweep is defined, run a single pass with [None]
        iterations = sweep_values if sweep_key else [None]

        for ds_index, (ds, model) in enumerate(zip(datasets, models)):
            self.console.log(f"Processing {ds.name}", color="blue")

            # Reference instances are drawn once per dataset, before the sweep, so
            # that every point of a curve is averaged over the same instances: a
            # variation along the curve then reflects the swept parameter alone.
            indices = self._draw_reference_instances(ds, config.n_runs, settings["seed"])
            n_runs = len(indices)

            for val in iterations:
                # Dynamically assign sweep parameter, otherwise fallback to config defaults
                k = val if sweep_key == "k" else config.k
                sigma = val if sweep_key == "sigma" else config.sigma
                m = val if sweep_key == "m" else config.m

                for run, idx in enumerate(indices):
                    msg = f"Running experiment {run + 1}/{n_runs}"
                    if sweep_key:
                        msg = f"{sweep_key.capitalize()} {val}: " + msg
                    self.console.log(msg)

                    x_test, y_test, Z, f_Z = self._prepare_run(
                        ds, model, ds_index, run, idx, sigma, m, settings["seed"]
                    )
                    explainers = self._build_explainers(
                        config.explainer_names,
                        mip_timeout=settings["mip_timeout"],
                        seed=settings["seed"],
                        sigma=sigma,
                        use_bias=settings["use_bias"],
                    )

                    run_metrics = eval_func(
                        x=x_test, y=y_test, Z=Z, f_Z=f_Z, maxsize=k,
                        features=ds.X.columns, model=model, sigma=sigma,
                        explainers=explainers, gen_method=settings["gen_method"],
                        eval_samples=settings["eval_samples"],
                    )

                    context = {
                        "dataset": ds.name,
                        "run": run,
                        "instance": int(idx),
                        "k": k,
                        "sigma": sigma,
                        "m": m,
                        "use_bias": settings["use_bias"],
                    }
                    
                    for metric in run_metrics:
                        metric.update(context)
                        if sweep_key:
                            metric.update({sweep_key: val})

                    all_results.extend(run_metrics)

        return self._save_results(all_results, datasets, prefix)

    def run_optimization_experiment(self, config: Any, is_classification: bool):
        return self._execute_experiment(
            config, is_classification, "Optimization Experiment", "optimization",
            self._evaluate_optimization_metrics
        )

    def run_generalization_experiment(self, config: Any, is_classification: bool):
        return self._execute_experiment(
            config, is_classification, "Generalization Experiment", "generalization",
            self._evaluate_generalization_metrics
        )

    def run_k_sweep_experiment(self, config: Any, is_classification: bool):
        return self._execute_experiment(
            config, is_classification, "Sparsity (k) Sweep", "k_sweep",
            self._evaluate_generalization_metrics, "k", config.k_range
        )

    def run_sigma_sweep_experiment(self, config: Any, is_classification: bool):
        return self._execute_experiment(
            config, is_classification, "Concentration (sigma) Sweep", "sigma_sweep",
            self._evaluate_generalization_metrics, "sigma", config.sigma_range
        )

    def run_m_sweep_experiment(self, config: Any, is_classification: bool):
        return self._execute_experiment(
            config, is_classification, "Sample Complexity (m) Sweep", "m_sweep",
            self._evaluate_generalization_metrics, "m", config.m_range
        )

    def run_convergence_experiment(self, config: Any, is_classification: bool):
        return self._execute_experiment(
            config, is_classification, "Convergence with (sigma) Sweep", "convergence",
            self._evaluate_optimization_metrics, "sigma", config.sigma_range
        )

    def run_time_sweep_experiment(self, config: Any, is_classification: bool):
        self.console.log("Running Time Sweep (MIP Incremental vs Baselines)", color="cyan")

        settings = self._settings(config)
        min_time = getattr(config, 'min_time', 10)
        max_time = getattr(config, 'max_time', 300)
        step_time = getattr(config, 'step_time', getattr(config, 'step', 10))
        sigma = getattr(config, "sigma", 1.0)

        datasets, models = self._prepare_pipeline(config.dataset_ids, is_classification, config)
        all_results = []

        time_steps = []
        curr_t = min_time
        while curr_t < max_time:
            time_steps.append(curr_t)
            curr_t += step_time
        time_steps.append(max_time)

        for ds_index, (ds, model) in enumerate(zip(datasets, models)):
            self.console.log(f"Processing {ds.name}", color="blue")

            indices = self._draw_reference_instances(ds, config.n_runs, settings["seed"])
            n_runs = len(indices)

            for run, idx in enumerate(indices):
                self.console.log(f"Starting Run {run + 1}/{n_runs} for {ds.name}")

                x_test, y_test, Z, f_Z = self._prepare_run(
                    ds, model, ds_index, run, idx, sigma, config.m, settings["seed"]
                )
                context = {
                    "dataset": ds.name, "run": run, "instance": int(idx),
                    "k": config.k, "sigma": sigma, "m": config.m,
                    "use_bias": settings["use_bias"],
                }

                # Baselines have no anytime behaviour: their single result is
                # replicated along the timeline as a flat reference curve.
                other_names = [name for name in config.explainer_names if name != "MIP"]
                if other_names:
                    explainers = self._build_explainers(
                        other_names,
                        mip_timeout=settings["mip_timeout"],
                        seed=settings["seed"],
                        sigma=sigma,
                        use_bias=settings["use_bias"],
                    )
                    run_metrics = self._evaluate_optimization_metrics(
                        x=x_test, y=y_test, Z=Z, f_Z=f_Z, maxsize=config.k,
                        features=ds.X.columns, model=model, explainers=explainers
                    )

                    for step in time_steps:
                        for metric in run_metrics:
                            row = metric.copy()
                            row.update(context)
                            row["expected_time"] = step
                            # is_optimal belongs to the MIP alone; marking the
                            # baselines True would merge two different meanings
                            # into one column.
                            row["is_optimal"] = None
                            all_results.append(row)

                if "MIP" in config.explainer_names:
                    explainer = MIPExplainer(
                        timeout=max_time, use_bias=settings["use_bias"], verbose=False
                    )
                    generator = explainer.explain_incremental(
                        x=x_test, y=y_test, Z=Z, f_Z=f_Z, maxsize=config.k,
                        features=ds.X.columns, min_time=min_time,
                        max_time=max_time, step=step_time
                    )

                    expected_time_step = min_time
                    for actual_walltime, explanation, _ in generator:
                        # The row is built by the shared helper, so the anytime
                        # curve carries the same columns as every other series.
                        row = self._metric_row(
                            "MIP", explainer, explanation, actual_walltime, x_test, y_test, Z, f_Z
                        )
                        row.update(context)
                        row["expected_time"] = expected_time_step
                        all_results.append(row)
                        expected_time_step = min(expected_time_step + step_time, max_time)

        return self._save_results(all_results, datasets, "time_sweep")