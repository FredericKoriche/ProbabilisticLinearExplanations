# -----------------------------------------------------------------------------
#
# Explainer Classes
#
# -----------------------------------------------------------------------------

import cvxpy as cp
import gurobipy as gp
import lime.lime_tabular
import numpy as np
import pandas as pd
import time
import torch

from abc import ABC, abstractmethod
from gurobipy import GRB
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from typing import Any, Generator, Tuple, Union, Optional
from utils import Console

# -----------------------------------------------------------------------------
# BaseExplainer
# Abstract base class for feature attribution explainers. Provides unified
# state management, logging, and RMSE evaluation for fidelity and relevance.
#
# An explanation is the affine function z -> w . z + w_0. The coefficients live
# in self.solution, aligned with the d columns of the dataset, and the bias in
# self.intercept, so that indices in self.solution keep matching `features`.
# Every metric below therefore evaluates through _predict rather than through a
# bare matrix product: forgetting the bias would silently shift the fidelity of
# any explainer that fits one, which is all of them.
# -----------------------------------------------------------------------------

class BaseExplainer(ABC):
    def __init__(self, verbose: bool = False):
        self.console = Console(verbose=verbose)
        self.explanation = {}
        self.walltime = float("inf")
        self.fidelity_error = float("inf")
        self.relevance_error = float("inf")
        self.solution = None
        self.intercept = 0.0
        self.solved = True
        # Signed distance to the anchoring hyperplane, recorded by
        # test_anchoring_violation next to the binary verdict.
        self.anchoring_gap = float("inf")
        self._name = self.__class__.__name__.replace("Explainer", "")

    def __str__(self) -> str:
        output = self.console.string(f"[{self._name}] Information", endl=True)
        output += self.console.string("Explanation", str(self.explanation), endl=True)
        output += self.console.string("Size", len(self.explanation), endl=True)
        output += self.console.string("Wall time", self.walltime, endl=True)
        output += self.console.string("Fidelity Error", self.fidelity_error, endl=True)
        output += self.console.string("Relevance Error", self.relevance_error, endl=True)
        return output

    def _get_solution_tensor(self, reference: torch.Tensor) -> torch.Tensor:
        """Helper to ensure the solution is consistently a torch.Tensor on the right device."""
        if isinstance(self.solution, torch.Tensor):
            return self.solution.to(dtype=reference.dtype, device=reference.device)
        return torch.tensor(self.solution, dtype=reference.dtype, device=reference.device)

    def _predict(self, Z: torch.Tensor) -> torch.Tensor:
        """Value of the affine explanation on a batch of instances, or on a single one."""
        sol_tensor = self._get_solution_tensor(Z)
        return Z @ sol_tensor + self.intercept

    @staticmethod
    def _as_vector(values: torch.Tensor) -> torch.Tensor:
        """
        Model outputs sometimes come back with a trailing singleton dimension.
        Left alone, subtracting a (n,) prediction from an (n, 1) target
        broadcasts into an (n, n) matrix and the loss is silently wrong.
        """
        return values.reshape(-1)

    def _generate_neighborhood(
        self, x: torch.Tensor, p: float, active_mask: torch.Tensor,
        use_exact: bool, num_samples: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Generates the evaluation neighborhood (either exact grid or Monte Carlo)."""
        d = x.shape[0]

        if not use_exact:
            Z_eval = x.repeat(num_samples, 1)
            flip_mask = torch.rand(num_samples, d, device=x.device) < p
            flip_mask[:, ~active_mask] = False
            Z_eval[flip_mask] *= -1
            return Z_eval, None

        unconstrained_dims = active_mask.sum().item()
        if unconstrained_dims == 0:
            return x.unsqueeze(0), torch.tensor([1.0], device=x.device)

        vals = torch.tensor([-1.0, 1.0], dtype=x.dtype, device=x.device)
        grids = torch.meshgrid(*[vals for _ in range(unconstrained_dims)], indexing='ij')
        combinations = torch.stack([g.flatten() for g in grids], dim=1)

        Z_eval = x.repeat(combinations.shape[0], 1)
        Z_eval[:, active_mask] = combinations

        flips = (combinations != x[active_mask]).float()
        h = flips.sum(dim=1)
        probs = (p ** h) * ((1.0 - p) ** (unconstrained_dims - h))
        return Z_eval, probs

    def _evaluate_metric(
        self, metric: str, x: torch.Tensor, model_predict: Any,
        p: float, active_mask: torch.Tensor, use_exact: bool, num_samples: int,
        f_x: Optional[Union[float, torch.Tensor]] = None
    ) -> float:
        """Unified method for computing both exact and approximate loss."""
        Z_eval, probs = self._generate_neighborhood(x, p, active_mask, use_exact, num_samples)

        with torch.no_grad():
            f_Z = self._as_vector(model_predict(Z_eval))

            if metric == "relevance":
                target = f_x
            else:  # fidelity
                target = self._predict(Z_eval)

            squared_diff = (f_Z - target) ** 2
            mse = torch.sum(probs * squared_diff) if use_exact else torch.mean(squared_diff)
            return float((mse / 4.0).item())

    def test_anchoring_violation(self, *, x: torch.Tensor, y: float, gamma: float = 1e-3) -> int:
        """Returns 1 if the explanation violates the anchoring constraint at x, else 0."""
        self.console.log(f"[{self._name}] Test anchoring constraint violation")
        if self.solution is None:
            return 1

        with torch.no_grad():
            prediction_at_x = self._predict(x)
            # The magnitude is kept alongside the verdict: a violation rate of 1
            # says nothing on its own, since missing the hyperplane by 1e-3 and
            # missing it by 0.5 are not the same failure.
            self.anchoring_gap = float(torch.abs(prediction_at_x - y).item())
            is_violated = int(self.anchoring_gap > gamma)

        self.console.log(f"[{self._name}] Anchoring Violation", is_violated)
        self.console.log(f"[{self._name}] Anchoring Gap", self.anchoring_gap)
        return is_violated

    def test_empirical_fidelity(self, *, Z_test: torch.Tensor, f_Z_test: torch.Tensor) -> float:
        self.console.log(f"[{self._name}] Test empirical fidelity")
        if self.solution is None:
            return float("inf")

        with torch.no_grad():
            predictions = self._predict(Z_test)
            targets = self._as_vector(f_Z_test)
            mse = torch.mean((predictions - targets) ** 2)
            self.fidelity_error = float((mse / 4.0).item())

        self.console.log(f"[{self._name}] Fidelity Error", self.fidelity_error)
        return self.fidelity_error

    def test_fidelity(
        self, *, x: torch.Tensor, sigma: float, model_predict: Any,
        method: str = "auto", exact_threshold: int = 18, num_samples: int = 500000
    ) -> float:
        self.console.log(f"[{self._name}] Test generative fidelity error")
        if self.solution is None:
            return float("inf")

        d = x.shape[0]
        active_mask = torch.ones(d, dtype=torch.bool, device=x.device)  # Flip all features
        p = np.exp(-sigma) / (1.0 + np.exp(-sigma))
        use_exact = (method == "exact") or (method == "auto" and d <= exact_threshold)

        self.console.log(f"[{self._name}] {'Exact' if use_exact else 'Monte Carlo'} evaluation")
        self.fidelity_error = self._evaluate_metric(
            "fidelity", x, model_predict, p, active_mask, use_exact, num_samples
        )

        self.console.log(f"[{self._name}] Fidelity Error", self.fidelity_error)
        return self.fidelity_error

    def test_relevance(
        self, *, x: torch.Tensor, f_x: Union[float, torch.Tensor], sigma: float,
        model_predict: Any, method: str = "auto", exact_threshold: int = 18,
        num_samples: int = 500000
    ) -> float:
        self.console.log(f"[{self._name}] Test relevance error")
        if self.solution is None:
            return float("inf")

        sol_tensor = self._get_solution_tensor(x)
        active_mask = torch.abs(sol_tensor) <= 1e-5  # Flip only non-support features

        unconstrained_dims = active_mask.sum().item()
        p = np.exp(-sigma) / (1.0 + np.exp(-sigma))
        use_exact = (method == "exact") or (method == "auto" and unconstrained_dims <= exact_threshold)

        self.console.log(f"[{self._name}] {'Exact' if use_exact else 'Monte Carlo'} evaluation")
        self.relevance_error = self._evaluate_metric(
            "relevance", x, model_predict, p, active_mask, use_exact, num_samples, f_x=f_x
        )

        self.console.log(f"[{self._name}] Relevance Error", self.relevance_error)
        return self.relevance_error

    @abstractmethod
    def explain(
        self, *, x: torch.Tensor, Z: torch.Tensor, f_Z: torch.Tensor, maxsize: int,
        features: pd.Index = None, **kwargs
    ) -> Tuple[dict, float]:
        pass
    
# -----------------------------------------------------------------------------
# CVXExplainer
# Convex optimization explainer utilizing Gurobi to minimize squared error
# under a bounded norm constraint.
#
# This is the continuous relaxation of the mixed-integer program: the
# combinatorial support constraint is replaced by ||w||_1 <= k. Since every
# admissible w satisfies ||w||_0 <= k and ||w||_infty <= 1, hence ||w||_1 <= k,
# the feasible set contains the admissible one and the optimal value is a valid
# lower bound on the optimum of the MIP. It is exported as self.fidelity_value,
# which is what makes it usable as a certificate when the MIP hits its time
# limit.
# -----------------------------------------------------------------------------

class CVXExplainer(BaseExplainer):
    def __init__(
        self,
        norm: Union[int, str] = 1,
        use_bias: bool = True,
        verbose: bool = False,
    ):
        super().__init__(verbose=verbose)
        self.norm = norm
        self.use_bias = use_bias
        self.status = "uninitialized"

        # Reporting
        self.intercept = 0.0
        self.solved = False
        self.inaccurate = False     # solver returned an inaccurate solution
        self.fidelity_value = None  # optimal value, in F_hat units

    def _augment(
        self, samples: np.ndarray, anchor: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        The constant coordinate is appended as a last column, so that w_0 is an
        ordinary variable: bounded by the box, counted by the norm constraint,
        and part of the anchoring hyperplane. Same convention as the MIP.
        """
        if not self.use_bias:
            return samples, anchor
        ones = np.ones((samples.shape[0], 1), dtype=samples.dtype)
        return np.hstack([samples, ones]), np.append(anchor, 1.0)

    def explain(
        self,
        *,
        x: torch.Tensor,
        y: float,
        Z: torch.Tensor,
        f_Z: torch.Tensor,
        maxsize: float,
        features: pd.Index = None,
        **kwargs,
    ) -> Tuple[dict, float]:
        self.console.log(f"[{self._name}] Find explanation")

        A_np = Z.detach().cpu().numpy()
        b_np = f_Z.detach().cpu().numpy().flatten()
        x_np = x.detach().cpu().numpy().flatten()

        self.solve(samples=A_np, labels=b_np, hyperplane=(x_np, y), maxsize=maxsize)

        self.explanation = {}
        if self.solved:
            threshold = 1e-5
            self.explanation = {
                (features[i] if features is not None else f"Feature_{i}"): float(w)
                for i, w in enumerate(self.solution)
                if abs(float(w)) > threshold
            }
            self.console.log(f"[{self._name}] Explanation Size", len(self.explanation))
            self.console.log(f"[{self._name}] Lower bound", self.fidelity_value)
            self.console.log(f"[{self._name}] Wall time", self.walltime)

        return self.explanation, self.walltime

    def solve(
        self,
        *,
        samples: np.ndarray,
        labels: np.ndarray,
        hyperplane: Tuple[np.ndarray, float],
        maxsize: float,
    ) -> None:
        self.console.log(f"[{self._name}] Solve problem")

        n_features = samples.shape[1]
        self.solved, self.inaccurate = False, False
        self.solution, self.intercept, self.fidelity_value = None, 0.0, None

        x, y = hyperplane
        A, x = self._augment(samples, x)
        b = labels
        m, n = A.shape
        k = maxsize
        q = self.norm

        # Model construction is timed with the solve. The Gram matrix below is
        # O(m d^2) and dominates the solve on the largest benchmarks, so leaving
        # it out would make the reported time incomparable to the baselines.
        time_start = time.time()
        try:
            # The box is declared through variable bounds rather than as 2n
            # explicit constraints: mathematically identical, but it keeps the
            # model at n rows instead of 3n, which the interior-point solver is
            # very sensitive to.
            w = cp.Variable(n, bounds=[-np.ones(n), np.ones(n)])
            constraints = [w @ x == y, cp.norm(w, q) <= k]

            # Minimizing ||A w - b||^2 is equivalent to minimizing
            # w' G w - 2 c' w, where G = A'A and c = A'b, the dropped constant
            # b'b being irrelevant to the argmin. This hands the solver an n x n
            # matrix instead of the m x n sample matrix, which is what makes the
            # problem tractable at large d. The constant is kept aside and added
            # back when the objective is converted into empirical fidelity.
            G = A.T @ A
            G = 0.5 * (G + G.T)  # symmetrize away the floating point drift
            c = A.T @ b
            offset = float(b @ b)

            objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(G)) - 2 * (c @ w))
            self.problem = cp.Problem(objective, constraints=constraints)

            # Using Gurobi to match MIP's single-threaded environment
            self.problem.solve(solver=cp.GUROBI, verbose=self.console.verbose, Threads=1)
            self.status = self.problem.status

            if self.status in ["optimal", "optimal_inaccurate"] and w.value is not None:
                self.solved = True
                self.inaccurate = self.status == "optimal_inaccurate"
                if self.inaccurate:
                    self.console.log(
                        f"[{self._name}] Solution reported as inaccurate: the "
                        "lower bound it certifies should not be relied upon",
                        color="yellow",
                    )

                full = np.asarray(w.value).flatten()
                if self.use_bias:
                    self.solution = full[:n_features].copy()
                    self.intercept = float(full[n_features])
                else:
                    self.solution = full.copy()
                    self.intercept = 0.0

                self.fidelity_value = (self.problem.value + offset) / (4.0 * m)
            else:
                self.console.log(
                    f"[{self._name}] Solver failed with status {self.status}", color="red"
                )

        except cp.error.SolverError as e:
            self.console.log(f"[{self._name}] Solver error: {e}", color="red")
            self.status = "error"
        finally:
            self.walltime = max(0.0, time.time() - time_start)
                        
# -----------------------------------------------------------------------------
# IHTExplainer
# Iterative Hard Thresholding explainer using the Greedy Selector and
# Hyperplane Projector (GSHP) for exact sparsity limits.
#
# The bias is carried as a constant coordinate appended to Z and to x. Since
# x_j = +1 there, the change of variable u = v * x used below is unaffected, and
# w_0 is projected, thresholded and counted exactly like any other coefficient,
# which is the convention of Section 3.
#
# The box ||w||_infty <= 1 is deliberately absent from the projection: GSHP is
# an exact Euclidean projection onto the sparsity set intersected with the
# anchoring hyperplane, and adding the box would break that exactness. The box
# is instead carried by the reference optimum in the analysis, and
# self.max_abs_weight records how far the iterates stay from it.
# -----------------------------------------------------------------------------

class IHTExplainer(BaseExplainer):
    def __init__(
        self,
        sigma: float = 0.0,
        iterations: int = 5000,
        epsilon: float = 1e-6,
        patience: int = 20,
        use_bias: bool = True,
        verbose: bool = False,
    ):
        super().__init__(verbose=verbose)
        self.sigma = sigma
        self.iterations = iterations
        self.epsilon = epsilon
        # Number of consecutive non-improving iterations tolerated before the
        # iterate is declared stationary (see the stopping rule in explain).
        self.patience = patience
        self.use_bias = use_bias

        # Reporting
        self.n_iterations = 0
        self.intercept = 0.0
        self.solved = False
        self.max_abs_weight = None
        self.fidelity_value = None  # loss of the returned iterate, in F_hat units
        self.monotone = True        # whether the last iterate was also the best

        # Calculate theoretical step size using sigma
        self.stepsize = (18.0 / 19.0) * (np.cosh(sigma / 2.0) ** 2)

        if self.console.verbose:
            self.console.log("[IHT] Theoretical step size (eta)", f"{self.stepsize:.4f}")

    def _gshp(self, u: torch.Tensor, k: int, y: float) -> torch.Tensor:
        """
        Greedy Selector and Hyperplane Projector (Kyrillidis et al.)
        Computes exact Euclidean projection in O(d log d + k^2).

        Selected coordinates are masked out with -inf rather than removed from an
        index list, which keeps every step on the device: the selection is
        identical, including tie-breaking, but no index is transferred back to
        the host inside the loop.
        """
        d = u.shape[0]
        neg_inf = torch.finfo(u.dtype).min

        selected = torch.zeros(d, dtype=torch.bool, device=u.device)
        S = torch.empty(k, dtype=torch.long, device=u.device)
        running_sum = torch.zeros((), dtype=u.dtype, device=u.device)

        for i in range(k):
            if i == 0:
                values = y * u
            else:
                avg = (running_sum - y) / i
                values = torch.abs(u - avg)

            idx = torch.argmax(values.masked_fill(selected, neg_inf))
            S[i] = idx
            selected[idx] = True
            running_sum = running_sum + u[idx]

        u_star = torch.zeros_like(u)
        tau = (running_sum - y) / k
        u_star[S] = u[S] - tau

        return u_star

    def explain(
        self,
        *,
        x: torch.Tensor,
        y: float,
        Z: torch.Tensor,
        f_Z: torch.Tensor,
        maxsize: int,
        features: pd.Index = None,
        **kwargs,
    ) -> Tuple[dict, float]:
        self.console.log(f"[{self._name}] Find explanation")

        time_start = time.time()

        f_Z = f_Z.flatten()
        x = x.flatten()
        if self.use_bias:
            ones = torch.ones(Z.shape[0], 1, dtype=Z.dtype, device=Z.device)
            Z = torch.cat([Z, ones], dim=1)
            x = torch.cat([x, ones[0]])

        m, d = Z.shape
        n_features = d - 1 if self.use_bias else d
        k = maxsize

        w = torch.zeros(d, dtype=Z.dtype, device=Z.device)
        w[0] = y * x[0]

        prev_loss = float("inf")
        best_loss = float("inf")
        best_w = w.clone()
        stalled = 0
        step = 0

        for step in range(self.iterations):
            err = Z @ w - f_Z
            current_loss = torch.mean(err ** 2).item()

            # The best iterate is kept rather than the last one. Under the
            # monotonicity used in the analysis the two coincide, so this costs
            # nothing; when they differ, self.monotone records it, which is a
            # cheap empirical check on the step size.
            if current_loss < best_loss:
                best_loss = current_loss
                best_w = w.clone()

            # Two ways of being done. Either two consecutive losses coincide, in
            # which case the iterate has settled; or the loss no longer improves
            # on the best value seen so far for `patience` iterations, in which
            # case hard thresholding is cycling between nearly equivalent
            # supports. The second case is not detectable from consecutive losses
            # alone: their difference stays above epsilon indefinitely while the
            # algorithm makes no further progress.
            if abs(prev_loss - current_loss) < self.epsilon:
                if self.console.verbose:
                    self.console.log(f"[{self._name}] Converged at iteration {step}")
                break

            if current_loss < best_loss - self.epsilon:
                stalled = 0
            else:
                stalled += 1
                if stalled >= self.patience:
                    if self.console.verbose:
                        self.console.log(
                            f"[{self._name}] Stalled at iteration {step} "
                            f"(no improvement over {self.patience} iterations)"
                        )
                    break

            prev_loss = current_loss

            grad = (1.0 / m) * Z.T @ err
            v = w - self.stepsize * grad

            u = v * x
            u_star = self._gshp(u, k, y)
            w = u_star * x

        # The loss of the final iterate is never seen inside the loop, so it is
        # evaluated once more before choosing what to return.
        final_loss = torch.mean((Z @ w - f_Z) ** 2).item()
        if final_loss < best_loss:
            best_loss, best_w = final_loss, w.clone()

        self.n_iterations = step + 1
        self.monotone = bool(np.isclose(final_loss, best_loss))
        if not self.monotone:
            self.console.log(
                f"[{self._name}] Last iterate worse than the best one "
                f"({final_loss:.6f} vs {best_loss:.6f}): returning the best",
                color="yellow",
            )

        w = best_w
        w[torch.abs(w) < 1e-6] = 0.0

        if self.use_bias:
            self.solution = w[:n_features]
            self.intercept = float(w[n_features].item())
        else:
            self.solution = w
            self.intercept = 0.0

        # Reported over the whole vector, bias included: it is the quantity the
        # box constraint bears on.
        self.max_abs_weight = float(w.abs().max().item())
        self.fidelity_value = best_loss / 4.0
        self.solved = True

        self.walltime = max(0.0, time.time() - time_start)

        threshold = 1e-5
        self.explanation = {
            (features[i] if features is not None else f"Feature_{i}"): float(weight.item())
            for i, weight in enumerate(self.solution)
            if abs(float(weight.item())) > threshold
        }

        self.console.log(f"[{self._name}] Iterations", self.n_iterations)
        self.console.log(f"[{self._name}] Max |w_j|", self.max_abs_weight)
        self.console.log(f"[{self._name}] Empirical fidelity", self.fidelity_value)
        self.console.log(f"[{self._name}] Explanation Size", len(self.explanation))
        self.console.log(f"[{self._name}] Wall time", self.walltime)

        return self.explanation, self.walltime
    
# -----------------------------------------------------------------------------
# LIMEExplainer
# Local Interpretable Model-agnostic Explanations tabular wrapper extracting
# linear coefficients from a localized surrogate model.
#
# LIME always fits an intercept, which Section 3 counts inside ||w||_0. Left at
# num_features = k, its explanations would hold k + 1 terms and its hypothesis
# space would strictly contain ours, which is why it is given a budget of k - 1
# here. The three spaces are then nested exactly:
#     W = B_0(k) inter H(x, f(x)) inter B_inf(1)   (MIP)
#     V = B_0(k) inter H(x, f(x))                  (IHT)
#     U = B_0(k)                                   (LIME)
# -----------------------------------------------------------------------------

class LIMEExplainer(BaseExplainer):
    def __init__(self, seed: int = 48, verbose: bool = False) -> None:
        super().__init__(verbose=verbose)
        self.model = None
        self.seed = seed

    def explain(
        self,
        *,
        x: torch.Tensor,
        Z: torch.Tensor,
        maxsize: int,
        features: pd.Index,
        f_Z: torch.Tensor = None,
        model: Any = None,
        **kwargs
    ) -> Tuple[dict, float]:
        self.console.log(f"[{self._name}] Find explanation")
        self.model = model

        training_data_np = Z.detach().cpu().numpy()
        target_np = x.detach().cpu().numpy()
        if target_np.ndim > 1:
            target_np = target_np.flatten()

        self.solve(
            features=features,
            maxsize=maxsize,
            samples=training_data_np,
            target=target_np
        )

        self.console.log(f"[{self._name}] Explanation")
        self.console.log(self.explanation)
        self.console.log(f"[{self._name}] Explanation Size", len(self.explanation))
        self.console.log(f"[{self._name}] Wall time", self.walltime)
        return self.explanation, self.walltime

    def _constant_explanation(
        self,
        *,
        features: pd.Index,
        samples: np.ndarray,
        predict: Any,
    ) -> None:
        """
        Degenerate case k = 1. With the bias counted inside ||w||_0, LIME is left
        with no coefficient at all and its surrogate reduces to the constant w_0.
        That constant is a well-defined member of the hypothesis space, but LIME
        cannot produce it: Ridge rejects a design matrix with zero columns. It is
        therefore built here, as the unweighted mean of the model over the
        neighborhood. LIME would weight that mean by its exponential kernel; the
        unweighted mean is the constant minimising empirical fidelity on the
        sample, hence the choice most favourable to the baseline.
        """
        time_start = time.time()
        predictions = predict(samples)
        self.walltime = max(0.0, time.time() - time_start)

        self.solution = np.zeros(len(features), dtype=float)
        self.intercept = float(np.mean(predictions))
        self.explanation = {}

    def solve(
        self,
        *,
        features: pd.Index,
        maxsize: int,
        samples: np.ndarray,
        target: np.ndarray,
    ) -> None:
        categorical_features = list(range(len(features)))

        self.lime = lime.lime_tabular.LimeTabularExplainer(
            samples,
            feature_names=list(features),
            class_names=["prediction"],
            mode="regression",
            categorical_features=categorical_features,
            discretize_continuous=False,
            random_state=self.seed,
            verbose=self.console.verbose,
        )

        def lime_predict_wrapper(z_array: np.ndarray) -> np.ndarray:
            preds_tensor = self.model.predict(z_array)
            return preds_tensor.detach().cpu().numpy().flatten()

        # One slot of the budget is reserved for the intercept LIME always fits,
        # so that its explanations hold k terms in the sense of Section 3.
        num_features = maxsize - 1
        if num_features < 1:
            self._constant_explanation(
                features=features, samples=samples, predict=lime_predict_wrapper
            )
            return

        time_start = time.time()
        result = self.lime.explain_instance(
            target,
            lime_predict_wrapper,
            num_features=num_features,
            num_samples=samples.shape[0],
        )
        time_end = time.time()
        self.walltime = max(0.0, time_end - time_start)

        # In regression mode LIME fills both keys, but stores under 0 the
        # negation of the weights it fitted:
        #     local_exp[1] = local_exp[0]  (the fitted weights)
        #     local_exp[0] = [(i, -j) for i, j in local_exp[1]]
        # Key 1 therefore carries the actual coefficients, and reading key 0
        # would flip every sign. The fallback covers the classification path,
        # where only the requested labels are present.
        explanation_map = result.as_map()
        label = 1 if 1 in explanation_map else 0
        feature_weights = explanation_map.get(label, [])

        # LIME regresses on the interpretable basis b_j = 1[z_j == x_j], whereas
        # our hypothesis space lives on z itself. Substituting
        # b_j = (1 + z_j x_j) / 2 turns the surrogate
        #     g(b) = c + sum_j v_j b_j
        # into
        #     g(z) = (c + sum_j v_j / 2) + sum_j (v_j x_j / 2) z_j,
        # which is the form used below.
        self.solution = np.zeros(len(features), dtype=float)
        self.intercept = float(result.intercept[label])

        for feat_idx, weight in feature_weights:
            self.solution[feat_idx] = 0.5 * weight * target[feat_idx]
            self.intercept += 0.5 * weight

        # At z = x every indicator b_j equals 1, so the rewritten surrogate must
        # reproduce the local prediction LIME reports at the reference instance.
        # The field changed shape across releases: older ones expose a plain
        # array, current ones a dict keyed by label -- and in regression mode
        # only label 0 is ever filled, since the loop runs on labels = [0].
        local_pred = result.local_pred
        if isinstance(local_pred, dict):
            local_pred = local_pred.get(0, next(iter(local_pred.values())))
        local_pred = float(np.ravel(local_pred)[0])

        rebuilt = self.intercept + float(self.solution @ target)
        if not np.isclose(rebuilt, local_pred, atol=1e-6):
            self.console.log(
                "LIME basis change is inconsistent with its local prediction",
                f"{rebuilt:.6f} vs {local_pred:.6f}",
                color="red",
            )

        self.explanation = {
            features[i]: float(self.solution[i])
            for i in range(len(features))
            if abs(self.solution[i]) > 1e-5
        }
        
# -----------------------------------------------------------------------------
# MAPLEExplainer
# Model Agnostic suPervised Local Explanations (Plumb et al., NeurIPS 2018).
#
# This is a port of the reference implementation released by the authors
# (https://github.com/GDPlumb/MAPLE, Code/MAPLE.py). It keeps both components of
# the method: SILO, which derives local neighborhood weights from random forest
# leaf co-occurrences, and DStump, which ranks features by the impurity at the
# root of each tree and selects how many of them to retain by validation.
#
# Deviations from the reference implementation, all documented in the paper:
#   1. The forest and the local models are fitted on Z ~ D_{x, sigma} rather
#      than on the training set of the black box, so that every explainer sees
#      the same local distribution.
#   2. The number of retained features is searched over a logarithmic grid
#      instead of every value in 1..d, which is intractable at d = 1652.
#   3. The validation set used by that search is capped, for the same reason.
# -----------------------------------------------------------------------------

class MAPLEExplainer(BaseExplainer):
    def __init__(
        self,
        n_estimators: int = 200,
        max_features: float = 0.5,
        min_samples_leaf: int = 10,
        regularization: float = 0.001,
        val_fraction: float = 0.2,
        max_val_points: int = 8,
        max_retain_values: int = 16,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        super().__init__(verbose=verbose)
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.regularization = regularization
        self.val_fraction = val_fraction
        self.max_val_points = max_val_points
        self.max_retain_values = max_retain_values
        self.seed = seed

        self.estimator = None
        self.train_leaf_ids = None
        self.intercept = 0.0
        self.retain = None

    # -- SILO ------------------------------------------------------------------

    def _training_point_weights(self, instance_leaf_ids: np.ndarray) -> np.ndarray:
        """
        Weight of every training point, as in Eq. 2 of the paper: for each tree,
        the points sharing the leaf of the instance split a unit of mass between
        them. Points in crowded leaves therefore weigh less than points in rare
        ones, which is what makes the neighborhood supervised rather than merely
        local.
        """
        matches = self.train_leaf_ids == instance_leaf_ids[None, :]
        counts = matches.sum(axis=0)
        counts = np.where(counts == 0, 1, counts)
        return (matches / counts).sum(axis=1)

    def _local_model(self, weights: np.ndarray, Z_sel: np.ndarray, f_Z: np.ndarray) -> Ridge:
        model = Ridge(alpha=self.regularization)
        model.fit(Z_sel, f_Z, sample_weight=weights)
        return model

    # -- DStump ----------------------------------------------------------------

    def _feature_scores(self, n_features: int) -> np.ndarray:
        """
        Non-normalized impurity at the root of each tree, accumulated on the
        feature that the root splits on.
        """
        scores = np.zeros(n_features, dtype=float)
        for tree in self.estimator.estimators_:
            splits = tree.tree_.feature  # -2 marks a leaf, index 0 is the root
            if splits[0] != -2:
                scores[splits[0]] += tree.tree_.impurity[0]
        return scores

    def _retain_grid(self, n_features: int) -> list:
        """
        The reference implementation sweeps every value in 1..d. That is
        quadratic in d once the inner validation loop is taken into account, so
        we sweep a logarithmic grid of at most max_retain_values values instead,
        always including 1 and d.
        """
        if n_features <= self.max_retain_values:
            return list(range(1, n_features + 1))
        grid = np.geomspace(1, n_features, num=self.max_retain_values)
        return sorted({int(round(v)) for v in grid} | {1, n_features})

    def _select_retain(
        self,
        Z_train: np.ndarray,
        f_train: np.ndarray,
        Z_val: np.ndarray,
        f_val: np.ndarray,
        ranking: np.ndarray,
    ) -> int:
        val_leaf_ids = self.estimator.apply(Z_val)
        val_weights = [self._training_point_weights(ids) for ids in val_leaf_ids]

        retain_best, rmse_best = 1, np.inf
        for retain in self._retain_grid(Z_train.shape[1]):
            selected = np.sort(ranking[:retain])
            Z_train_p = Z_train[:, selected]
            Z_val_p = Z_val[:, selected]

            predictions = np.empty(Z_val.shape[0], dtype=float)
            for i, weights in enumerate(val_weights):
                model = self._local_model(weights, Z_train_p, f_train)
                predictions[i] = model.predict(Z_val_p[i].reshape(1, -1))[0]

            rmse = np.sqrt(np.mean((predictions - f_val) ** 2))
            if rmse < rmse_best:
                rmse_best, retain_best = rmse, retain

        return retain_best

    # -- Explainer interface ---------------------------------------------------

    def explain(
        self,
        *,
        x: torch.Tensor,
        Z: torch.Tensor,
        f_Z: torch.Tensor,
        maxsize: int,
        features: pd.Index = None,
        **kwargs,
    ) -> Tuple[dict, float]:
        # maxsize is accepted for interface uniformity but deliberately unused:
        # MAPLE selects the size of its own support and takes no sparsity budget.
        self.console.log(f"[{self._name}] Find explanation")

        Z_np = Z.detach().cpu().numpy()
        f_Z_np = f_Z.detach().cpu().numpy().flatten()
        x_np = x.detach().cpu().numpy().flatten()
        n_samples, n_features = Z_np.shape

        time_start = time.time()

        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n_samples)
        n_val = min(self.max_val_points, max(1, int(self.val_fraction * n_samples)))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        Z_train, f_train = Z_np[train_idx], f_Z_np[train_idx]
        Z_val, f_val = Z_np[val_idx], f_Z_np[val_idx]

        self.estimator = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.seed,
        )
        self.estimator.fit(Z_train, f_train)
        self.train_leaf_ids = self.estimator.apply(Z_train)

        ranking = np.argsort(-self._feature_scores(n_features))
        self.retain = self._select_retain(Z_train, f_train, Z_val, f_val, ranking)
        selected = np.sort(ranking[: self.retain])

        weights = self._training_point_weights(self.estimator.apply([x_np])[0])
        model = self._local_model(weights, Z_train[:, selected], f_train)

        self.solution = np.zeros(n_features, dtype=float)
        self.solution[selected] = model.coef_
        self.intercept = float(model.intercept_)

        time_end = time.time()
        self.walltime = max(0.0, time_end - time_start)

        threshold = 1e-5
        self.explanation = {
            (features[i] if features is not None else f"Feature_{i}"): float(w)
            for i, w in enumerate(self.solution)
            if abs(float(w)) > threshold
        }

        self.console.log(f"[{self._name}] Retained features", self.retain)
        self.console.log(f"[{self._name}] Explanation Size", len(self.explanation))
        self.console.log(f"[{self._name}] Wall time", self.walltime)

        return self.explanation, self.walltime
    
# -----------------------------------------------------------------------------
# MIPExplainer
# Mixed-Integer Programming explainer applying Gurobi solver to find an
# optimal minimal-error subset under hard cardinality constraints.
#
# The sparsity constraint is encoded with indicator variables and a big-M of 1,
# which is exact here because the box constraint already bounds |w_j| by 1. This
# is the same program as the SOS1 encoding, with a tighter linear relaxation.
#
# The solver works on the unnormalized objective sum_i (w . z_i - f(z_i))^2,
# which equals 4 * m * F_hat(w). Reported values are converted back to the
# normalized empirical fidelity of the paper through self.objective_scale.
# -----------------------------------------------------------------------------

class MIPExplainer(BaseExplainer):
    def __init__(self, timeout: int = 10, use_bias: bool = True, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.timeout = timeout
        self.use_bias = use_bias
        self.status = "uninitialized"
        self.problem = None
        self._env = None
        self._w = None

        # Reporting
        self.objective_scale = 1.0
        self.intercept = 0.0
        self.solved = False        # a usable incumbent was returned
        self.is_optimal = False    # optimality was proven within the time limit
        self.mip_gap = None        # relative gap, None when no incumbent
        self.fidelity_value = None # incumbent objective, in F_hat units
        self.fidelity_bound = None # dual bound, in F_hat units
        self.optimal_time = None   # the exact moment optimality is proven

    # -- Model -----------------------------------------------------------------

    def _augment(
        self, samples: np.ndarray, anchor: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        With a bias term, the constant coordinate is appended as a last column,
        so that w_0 is an ordinary variable: it is bounded by the box, counted by
        the cardinality constraint, and part of the anchoring hyperplane.
        """
        if not self.use_bias:
            return samples, anchor
        ones = np.ones((samples.shape[0], 1), dtype=samples.dtype)
        return np.hstack([samples, ones]), np.append(anchor, 1.0)

    def _build_model(
        self,
        *,
        samples: np.ndarray,
        labels: np.ndarray,
        hyperplane: Tuple[np.ndarray, float],
        maxsize: int,
    ) -> None:
        self.console.log(f"[{self._name}] Build problem")

        x, y = hyperplane
        A, x = self._augment(samples, x)
        b = labels
        m, n = A.shape
        l = max(0, n - maxsize)

        self.objective_scale = 1.0 / (4.0 * m)

        self._env = gp.Env(empty=True)
        if not self.console.verbose:
            self._env.setParam("OutputFlag", 0)
        self._env.start()

        self.problem = gp.Model("mip_explainer", env=self._env)
        self.problem.setParam(GRB.Param.Threads, 1)

        s = self.problem.addMVar(n, vtype=GRB.BINARY, name="s")
        self._w = self.problem.addMVar(n, vtype=GRB.CONTINUOUS, lb=-1.0, ub=1.0, name="w")

        self.problem.addConstr(s.sum() >= l, name="cd_cons")
        self.problem.addConstr(self._w @ x == y, name="ln_cons")

        # s_j = 1 forces w_j = 0; s_j = 0 leaves w_j free in [-1, 1]. Exact
        # because the box already provides the big-M.
        self.problem.addConstr(self._w <= 1 - s, name="ub_cons")
        self.problem.addConstr(self._w >= s - 1, name="lb_cons")

        diff = A @ self._w - b
        self.problem.setObjective(diff @ diff, GRB.MINIMIZE)
        self.problem.update()

    def _collect(self, n_features: int) -> None:
        """
        Read the incumbent and the optimality information out of the model. Must
        run before the environment is disposed of.
        """
        self.solved = self.problem.SolCount > 0

        if self.solved:
            full = self._w.X
            if self.use_bias:
                self.solution = full[:n_features].copy()
                self.intercept = float(full[n_features])
            else:
                self.solution = full.copy()
                self.intercept = 0.0
            self.fidelity_value = float(self.problem.ObjVal) * self.objective_scale
        else:
            self.solution = None
            self.intercept = 0.0
            self.fidelity_value = None

        self.is_optimal = self.status == GRB.OPTIMAL
        try:
            self.fidelity_bound = float(self.problem.ObjBound) * self.objective_scale
            self.mip_gap = float(self.problem.MIPGap) if self.solved else None
        except (AttributeError, gp.GurobiError):
            self.fidelity_bound, self.mip_gap = None, None

    def _dispose(self) -> None:
        for handle in (self.problem, self._env):
            try:
                handle.dispose()
            except (AttributeError, gp.GurobiError):
                pass
        self.problem, self._env, self._w = None, None, None

    def _as_explanation(self, features: pd.Index = None) -> dict:
        if not self.solved:
            return {}
        threshold = 1e-5
        return {
            (features[i] if features is not None else f"Feature_{i}"): float(w)
            for i, w in enumerate(self.solution)
            if abs(float(w)) > threshold
        }

    # -- Explainer interface ---------------------------------------------------

    def explain(
        self,
        *,
        x: torch.Tensor,
        y: float,
        Z: torch.Tensor,
        f_Z: torch.Tensor,
        maxsize: int,
        features: pd.Index = None,
        **kwargs,
    ) -> Tuple[dict, float]:
        self.console.log(f"[{self._name}] Find explanation")

        A_np = Z.detach().cpu().numpy()
        b_np = f_Z.detach().cpu().numpy().flatten()
        x_np = x.detach().cpu().numpy().flatten()

        self.solve(samples=A_np, labels=b_np, hyperplane=(x_np, y), maxsize=maxsize)

        self.explanation = self._as_explanation(features)
        if self.solved:
            self.console.log(f"[{self._name}] Explanation Size", len(self.explanation))
            self.console.log(f"[{self._name}] Empirical fidelity", self.fidelity_value)
            self.console.log(f"[{self._name}] Proven optimal", self.is_optimal)
            self.console.log(f"[{self._name}] Wall time", self.walltime)

        return self.explanation, self.walltime

    def solve(
        self,
        *,
        samples: np.ndarray,
        labels: np.ndarray,
        hyperplane: Tuple[np.ndarray, float],
        maxsize: int,
    ) -> None:
        self.console.log(f"[{self._name}] Solve problem")

        n_features = samples.shape[1]
        self.solved, self.is_optimal = False, False
        self.solution, self.intercept = None, 0.0
        self.mip_gap, self.fidelity_value, self.fidelity_bound = None, None, None

        # Model construction is timed with the solve, so that the reported time
        # covers the same work as the wall time of the other explainers.
        time_start = time.time()
        try:
            self._build_model(
                samples=samples, labels=labels, hyperplane=hyperplane, maxsize=maxsize
            )
            self.problem.setParam(GRB.Param.TimeLimit, self.timeout)
            self.problem.optimize()
            self.status = self.problem.Status
            self._collect(n_features)

            if not self.solved:
                reason = (
                    "time limit reached with no feasible solution"
                    if self.status == GRB.TIME_LIMIT
                    else f"status code {self.status}"
                )
                self.console.log(f"[{self._name}] Solver stopped: {reason}", color="red")

        except gp.GurobiError as e:
            self.console.log(f"[{self._name}] Error code {e.errno}: {e}", color="red")
            self.status = "error"
        except AttributeError as e:
            self.console.log(f"[{self._name}] Attribute error: {e}", color="red")
            self.status = "error"
        finally:
            self._dispose()
            self.walltime = max(0.0, time.time() - time_start)

    def explain_incremental(
        self,
        *,
        x: torch.Tensor,
        y: float,
        Z: torch.Tensor,
        f_Z: torch.Tensor,
        maxsize: int,
        features: pd.Index = None,
        min_time: int = 10,
        max_time: int = 300,
        step: int = 10,
        **kwargs,
    ) -> Generator[Tuple[float, dict, bool], None, None]:
        self.console.log(f"[{self._name}] Setup incremental explanation")
        self.optimal_time = None

        A_np = Z.detach().cpu().numpy()
        b_np = f_Z.detach().cpu().numpy().flatten()
        x_np = x.detach().cpu().numpy().flatten()
        n_features = A_np.shape[1]

        time_start = time.time()
        self._build_model(
            samples=A_np, labels=b_np, hyperplane=(x_np, y), maxsize=maxsize
        )

        try:
            cumulative_time = time.time() - time_start
            current_step_limit = min_time

            while cumulative_time < max_time:
                self.problem.setParam(GRB.Param.TimeLimit, current_step_limit)

                self.console.log(
                    f"[{self._name}] Solving... (Running chunk: {current_step_limit}s)"
                )
                self.problem.optimize()
                self.status = self.problem.Status
                self._collect(n_features)

                cumulative_time = time.time() - time_start
                self.walltime = cumulative_time
                self.explanation = self._as_explanation(features)

                if self.is_optimal and self.optimal_time is None:
                    self.optimal_time = cumulative_time
                    self.console.log(
                        f"[{self._name}] Optimal solution found at {self.optimal_time:.1f}s."
                    )

                yield self.walltime, self.explanation, self.is_optimal

                # Pad the remaining timeline if the optimum is found early
                if self.is_optimal:
                    self.console.log(f"[{self._name}] Padding timeline to {max_time}s.")
                    while cumulative_time < max_time:
                        cumulative_time += min(step, max_time - cumulative_time)
                        self.walltime = cumulative_time
                        yield self.walltime, self.explanation, True
                    break

                current_step_limit = min(step, max_time - cumulative_time)
                if current_step_limit <= 0:
                    break
        finally:
            self._dispose()