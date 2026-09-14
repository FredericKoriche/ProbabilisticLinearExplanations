# -----------------------------------------------------------------------------
#
# Displayer Class
#
# Tables are laid out with (dataset, explainer) as rows and metrics as columns.
# The previous layout put every metric-explainer pair in its own column, which
# reached thirty columns and could not fit a page; forty rows of six columns do.
#
# -----------------------------------------------------------------------------

import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path


class Displayer:
    # Solid for the exact solver, densely dotted for IHT on top of it: where the
    # two coincide -- which series 1 and 2 show they nearly always do -- the
    # superposition is visible rather than hidden.
    STYLES = {
        "MIP": ((1, 0), "o"),
        "IHT": ((1, 1), "s"),
        "LIME": ((4, 2), "X"),
        "MAPLE": ((6, 2, 1, 2), "^"),
        "CVX": ((2, 2), "D"),
    }

    # The competitor carries the magenta of the triad; the two methods of the
    # paper carry the blue and the green, so that they read as a pair against
    # it. The dense baselines are greyed, being reported but not compared.
    COLORS = {
        "MIP": "#3E65FE",
        "IHT": "#48B52D",
        "LIME": "#FE3E65",
        "MAPLE": "#F5A623",
        "CVX": "#8E6FD8",
    }

    # Boolean dimension of each benchmark after preprocessing, used to order the
    # colour gradient of the anchoring figure: the colour then carries an
    # attribute of the benchmark instead of merely separating the series.
    DIMENSIONS = {
        "Medical Charges": 12, "Credit Approval": 17, "COMPAS": 20,
        "Postoperative": 23, "California Housing": 32, "Heart Disease": 37,
        "Forest Fires": 40, "Adult": 42, "Seoul Bike Sharing": 46,
        "Student Performance": 69, "Nomao": 149, "Speed Dating": 180,
        "NCI 60 Thioguanine": 188, "Communities and Crime": 367,
        "Musk": 644, "Cancer Drug Response Methylation": 1652,
    }

    def __init__(self, output_dir: str = "./exps"):
        self.output_dir = Path(output_dir)
        (self.output_dir / "tables").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(parents=True, exist_ok=True)

        # AIJ Custom Triad: Blue, Magenta, Green
        custom_triad = ["#3E65FE", "#FE3E65", "#65FE3E"]
        sns.set_theme(style="whitegrid", palette=custom_triad, context="paper", font_scale=1.2)

        # For LaTeX formatting
        matplotlib.rcParams.update({'text.usetex': True})
        matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amsfonts,amssymb,bm}'
        plt.rcParams.update({'text.usetex': True})

    # -- Helpers ---------------------------------------------------------------

    def _get_timestamp(self) -> str:
        return time.strftime("%Y%m%d-%H%M%S")

    def _save_pdf(self, filename: str, also_png: bool = True):
        """
        The PDF is what goes to the publisher; the PNG is a working copy, since
        a vector figure cannot be looked at without a viewer.
        """
        path = self.output_dir / "plots" / f"{filename}.pdf"
        plt.savefig(path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to: {path}")

        if also_png:
            preview = path.with_suffix(".png")
            plt.savefig(preview, bbox_inches='tight', dpi=150)

    def _save_tex(self, df: pd.DataFrame, filename: str):
        path = self.output_dir / "tables" / f"{filename}.tex"
        try:
            tex_str = df.style.to_latex(hrules=True, multirow_align="c")
            with open(path, "w") as f:
                f.write(tex_str)
        except AttributeError:
            df.to_latex(path, index=True, escape=False)
        print(f"Table saved to: {path}")

    def _style_args(self, explainers: list) -> dict:
        """Per-explainer dashes and markers, in the order seaborn will use them."""
        order = [e for e in ["MIP", "IHT", "LIME", "MAPLE", "CVX"] if e in explainers]
        return {
            "hue_order": order,
            "style_order": order,
            "dashes": [self.STYLES[e][0] for e in order],
            "markers": [self.STYLES[e][1] for e in order],
        }

    def _plot_dataset_loop(self, df: pd.DataFrame, prefix: str, plot_func):
        """Abstracts the boilerplate for dataset iteration and saving."""
        ts = self._get_timestamp()
        for ds in df['dataset'].unique():
            ds_df = df[df['dataset'] == ds].copy()
            plt.figure(figsize=(8, 5))

            plot_func(ds_df, str(ds))

            sns.despine()
            safe_name = str(ds).replace(" ", "_").lower()
            self._save_pdf(f"{prefix}_{safe_name}_{ts}")
            plt.close()

    @staticmethod
    def _pm(stats: pd.DataFrame, metric: str, digits: int = 3) -> pd.Series:
        """Formats mean +/- std, or a dash when the metric does not apply."""
        def fmt(row):
            m, s = row[(metric, 'mean')], row[(metric, 'std')]
            if pd.isna(m):
                return "-"
            if pd.isna(s):
                s = 0.0
            return rf"${m:.{digits}f} \pm {s:.{digits}f}$"
        return stats.apply(fmt, axis=1)

    # -- Tables ----------------------------------------------------------------

    def generate_optimization_table(self, csv_path: str):
        """
        Series 1-2. One row per (dataset, explainer). The anchoring gap sits next
        to its rate: a rate of 1.000 says nothing on its own, since missing the
        hyperplane by 1e-3 and missing it by 0.5 are not the same failure.
        """
        df = pd.read_csv(csv_path)
        metrics = ['emp_fidelity', 'sparsity', 'anchoring_violation',
                   'anchoring_gap', 'max_abs_weight', 'walltime', 'is_optimal']
        available = [m for m in metrics if m in df.columns]
        df[available] = df[available].apply(pd.to_numeric, errors='coerce')

        # 'max' is needed by the anchoring gap below: its distribution is very
        # skewed, so the worst case over the runs is more informative than a
        # dispersion, which would suggest a symmetry that is not there.
        stats = df.groupby(['dataset', 'explainer'])[available].agg(['mean', 'std', 'max'])
        table = pd.DataFrame(index=stats.index)

        table[r'$\widehat{\mathsf{F}}$'] = self._pm(stats, 'emp_fidelity')
        table[r'Size ($L_0$)'] = self._pm(stats, 'sparsity', digits=1)
        if 'anchoring_gap' in available:
            # The max is over the ten runs of the series, so it is the worst
            # case observed and not a bound.
            gap = stats['anchoring_gap']
            table[r'Anch. Gap (mean / max)'] = gap.apply(
                lambda row: "-" if pd.isna(row['mean'])
                else rf"${row['mean']:.3f}$ / ${row['max']:.3f}$", axis=1
            )
        # A rate over the runs, so no dispersion is reported: a std of 0.422 on
        # ten runs only encodes "2 out of 10" and reads as if the quantity were
        # symmetric around its mean.
        violation = stats[('anchoring_violation', 'mean')]
        table[r'Anch. Viol.'] = violation.map(
            lambda v: "-" if pd.isna(v) else rf"${v:.2f}$"
        )
        if 'max_abs_weight' in available:
            # The box is a constraint of W, so how far the unconstrained
            # baselines sit outside it is what says whether it binds at all.
            table[r'$\max_j |w_j|$'] = self._pm(stats, 'max_abs_weight')
        table[r'Time (s)'] = self._pm(stats, 'walltime')
        if 'is_optimal' in available:
            # Proven-optimal rate, which is what licenses the word "exact".
            # Blank for every explainer that does not certify anything.
            proven = stats[('is_optimal', 'mean')]
            table[r'Opt.'] = proven.map(lambda v: "-" if pd.isna(v) else rf"${v:.2f}$")

        self._save_tex(table, f"optimization_{self._get_timestamp()}")
        return table

    def generate_optimization_diagnostics(self, csv_path: str):
        """Appendix companion: quantities only one explainer reports."""
        df = pd.read_csv(csv_path)
        metrics = [m for m in ['n_iterations', 'mip_gap',
                               'solver_fidelity', 'retain'] if m in df.columns]
        df[metrics] = df[metrics].apply(pd.to_numeric, errors='coerce')

        stats = df.groupby(['dataset', 'explainer'])[metrics].agg(['mean', 'std'])
        table = pd.DataFrame(index=stats.index)
        labels = {
            'n_iterations': r'Iterations',
            'mip_gap': r'MIP Gap',
            'solver_fidelity': r'Solver $\widehat{\mathsf{F}}$',
            'retain': r'Retained',
        }
        for metric in metrics:
            digits = 1 if metric in ('n_iterations', 'retain') else 3
            table[labels[metric]] = self._pm(stats, metric, digits=digits)

        # Rows that carry no diagnostic at all are dropped.
        table = table[(table != "-").any(axis=1)]
        self._save_tex(table, f"optimization_diagnostics_{self._get_timestamp()}")
        return table

    def generate_generalization_table(self, csv_path: str):
        """
        Series 3-4. Empirical fidelity is kept alongside the population metrics:
        the gap between the first two columns is the generalization error, and
        relevance is what anchoring buys.
        """
        df = pd.read_csv(csv_path)
        metrics = [m for m in ['emp_fidelity', 'fidelity', 'relevance',
                               'anchoring_gap'] if m in df.columns]
        df[metrics] = df[metrics].apply(pd.to_numeric, errors='coerce')

        stats = df.groupby(['dataset', 'explainer'])[metrics].agg(['mean', 'std'])
        table = pd.DataFrame(index=stats.index)
        table[r'$\widehat{\mathsf{F}}$'] = self._pm(stats, 'emp_fidelity')
        table[r'Fidelity ($\mathsf{F}$)'] = self._pm(stats, 'fidelity')
        table[r'Relevance ($\mathsf{R}$)'] = self._pm(stats, 'relevance')
        if 'anchoring_gap' in metrics:
            # Carried over from the optimization table: relevance measures the
            # deviation from the constant f(x), so an explanation that misses
            # that value at x pays for it here.
            table[r'Anch. Gap'] = self._pm(stats, 'anchoring_gap')

        self._save_tex(table, f"generalization_{self._get_timestamp()}")
        return table

    def generate_support_overlap(self, csv_path: str, reference: str = "MIP"):
        """
        How much of the reference support each explainer recovers, run by run.
        Relevance is governed by which coordinates are held fixed, so an
        explainer can lose on relevance either because it misses the anchoring
        value or because it selects a different support; this table separates
        the two.

        Read alongside the reference against itself: where several supports
        achieve the same fidelity, a low overlap does not by itself mean a worse
        choice, and the IHT row -- which does match the reference optimum -- is
        the control that says whether the support is identifiable at all.
        """
        df = pd.read_csv(csv_path)
        if 'support' not in df.columns:
            raise ValueError(f"Missing 'support' column in {csv_path}.")

        supports = (
            df.set_index(['dataset', 'run', 'explainer'])['support']
              .map(lambda s: set(str(s).split("|")) - {"", "nan"})
        )

        rows = []
        for (dataset, run), group in supports.groupby(level=[0, 1]):
            target = group.get((dataset, run, reference))
            if not target:
                continue
            for (_, _, explainer), support in group.items():
                rows.append({
                    "dataset": dataset,
                    "explainer": explainer,
                    "overlap": len(support & target) / len(target),
                })

        overlap = pd.DataFrame(rows)
        stats = overlap.groupby(['dataset', 'explainer'])[['overlap']].agg(['mean', 'std'])
        table = pd.DataFrame(index=stats.index)
        table[rf'Support overlap with {reference}'] = self._pm(stats, 'overlap')

        self._save_tex(table, f"support_overlap_{self._get_timestamp()}")
        return table

    # -- Figures ---------------------------------------------------------------

    def generate_k_sweep_plot(self, csv_path: str):
        """
        Series 5-6. Relevance against the budget: empirical fidelity decreases
        with k by construction, so only relevance arbitrates a trade-off, and the
        exponential factor of the bound predicts an interior optimum.

        Three explainers only. The dense ones are dropped after series 3-4: the
        relevance neighbourhood frees the coordinates outside the support, so an
        explanation holding hundreds of them leaves almost nothing to vary and
        its relevance is low for a reason that has nothing to do with quality.
        Relevance is comparable at comparable support size, which is exactly
        what a budget sweep holds fixed.

        Grouped bars rather than curves: the exact solver and IHT reach the same
        relevance at almost every budget, and two lines that keep crossing read
        as noise. Side by side, equal heights read as agreement, which is the
        point. No dispersion is drawn -- four panels of three series each would
        disappear under the bars; the reference instances are common to the
        three explainers, so the comparison is paired and the text reports it.
        """
        df = pd.read_csv(csv_path)

        def _plot(ds_df, ds_name):
            order = [e for e in ["MIP", "IHT", "LIME", "MAPLE", "CVX"]
                     if e in set(ds_df["explainer"])]
            budgets = sorted(ds_df["k"].unique())
            positions = {b: i for i, b in enumerate(budgets)}

            width = 0.8 / len(order)

            for index, explainer in enumerate(order):
                mean = ds_df[ds_df["explainer"] == explainer].groupby("k")["relevance"].mean()
                offset = (index - (len(order) - 1) / 2) * width
                plt.bar(
                    [positions[b] + offset for b in mean.index], mean.values,
                    width=width * 0.92, label=explainer,
                    color=self.COLORS[explainer], edgecolor="white", linewidth=0.6,
                    zorder=3,
                )

            plt.xticks(range(len(budgets)), [str(b) for b in budgets])
            plt.xlabel(r"Sparsity Budget ($k$)")
            plt.ylabel(r"Relevance Error ($\mathsf{R}$)")
            plt.legend(title="Explainer", loc="upper right", framealpha=1.0)
            plt.grid(axis="x", visible=False)

        self._plot_dataset_loop(df, "k_sweep", _plot)

    def generate_anchoring_correlation_plot(
        self, csv_path: str, baseline: str = "LIME", reference: str = "MIP",
        relative: bool = True, floor: float = 0.0, highlight: int = 1,
        label_offset: tuple = (-10, 12),
    ):
        """
        Series 1-2. How much empirical fidelity the unanchored baseline gains
        against the exact solver, as a function of how far it misses the
        anchoring hyperplane. One point per run.

        One panel per CSV, so one per task: the anchoring gap lives on the scale
        of the model output, which differs between a classifier valued in
        {-1,+1} and a regressor on concentrated targets. Pooling the two tasks
        destroys the relation, so they are never drawn together.

        Runs of one benchmark share a colour, and the colours are graded by the
        Boolean dimension of the benchmark, so the gradient answers on its own
        the question the reader asks next -- whether the relation is driven by
        dimension. A colour bar replaces a legend that eight names would make
        unreadable at half width.
        """
        df = pd.read_csv(csv_path)
        keys = ["dataset", "run"]
        base = df[df["explainer"] == baseline].set_index(keys)
        ref = df[df["explainer"] == reference].set_index(keys)

        points = pd.DataFrame({
            "gap": base["anchoring_gap"],
            "base_fidelity": base["emp_fidelity"],
            "ref_fidelity": ref["emp_fidelity"],
        }).dropna().reset_index()

        if relative:
            points = points[points["ref_fidelity"] >= floor]
            points["advantage"] = (
                1.0 - points["base_fidelity"] / points["ref_fidelity"]
            )
            ylabel = (rf"$1 - \widehat{{\mathsf{{F}}}}"
                      rf"(\bm{{w}}_{{\mathrm{{{baseline}}}}})\,/\,"
                      rf"\widehat{{\mathsf{{F}}}}"
                      rf"(\bm{{w}}_{{\mathrm{{{reference}}}}})$")
        else:
            points["advantage"] = points["ref_fidelity"] - points["base_fidelity"]
            ylabel = (rf"$\widehat{{\mathsf{{F}}}}"
                      rf"(\bm{{w}}_{{\mathrm{{{reference}}}}}) - "
                      rf"\widehat{{\mathsf{{F}}}}"
                      rf"(\bm{{w}}_{{\mathrm{{{baseline}}}}})$")

        if points.empty:
            raise ValueError(f"No comparable runs between {baseline} and {reference}.")

        # Slope and correlation are accumulated from centred products rather
        # than obtained from polyfit, corrcoef, cov or Series.corr: every one of
        # those routes through a matrix product.
        gx, gy = points["gap"], points["advantage"]
        cx, cy = gx - gx.mean(), gy - gy.mean()
        sxy, sxx, syy = float((cx * cy).sum()), float((cx * cx).sum()), float((cy * cy).sum())
        slope = sxy / sxx if sxx else 0.0
        intercept = gy.mean() - slope * gx.mean()
        r = sxy / ((sxx * syy) ** 0.5) if sxx and syy else 0.0

        plt.figure(figsize=(7, 5))
        plt.axhline(0.0, color="gray", linewidth=0.8, zorder=0)

        present = sorted(points["dataset"].unique(),
                         key=lambda ds: self.DIMENSIONS.get(ds, 0))
        dims = [self.DIMENSIONS.get(ds, 1) for ds in present]
        norm = matplotlib.colors.LogNorm(vmin=min(dims), vmax=max(dims))
        cmap = matplotlib.colormaps["coolwarm"]

        for dataset, dimension in zip(present, dims):
            subset = points[points["dataset"] == dataset]
            plt.scatter(subset["gap"], subset["advantage"], s=34, alpha=0.85,
                        color=cmap(norm(dimension)), edgecolor="white",
                        linewidth=0.3, zorder=3)

        span = [float(gx.min()), float(gx.max())]
        plt.plot(span, [slope * span[0] + intercept, slope * span[1] + intercept],
                 color="0.25", linestyle="--", linewidth=1.2, zorder=1)
        plt.annotate(rf"$r = {r:.2f},\ n = {len(points)}$", xy=(0.04, 0.93),
                     xycoords="axes fraction", fontsize="medium")

        # # The widest-gap benchmark is named in place, so that the colour bar has
        # # at least one anchor in the reader's memory.
        # named = [points.groupby("dataset")["gap"].mean().idxmax()] if highlight else []
        # for dataset in named:
        #     subset = points[points["dataset"] == dataset]
        #     top = subset.loc[subset["advantage"].idxmax()]
        #     plt.annotate(
        #         dataset, xy=(top["gap"], top["advantage"]),
        #         xytext=label_offset, textcoords="offset points", fontsize="small",
        #         ha="right" if label_offset[0] < 0 else "left", zorder=4,
        #     )

        bar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                           ax=plt.gca(), pad=0.02)
        bar.set_label(r"Dimension $d$")

        plt.xlabel(rf"Anchoring gap $|\bm{{w}} \cdot \bm{{x}} - f(\bm{{x}})|$ "
                   rf"of {baseline}")
        plt.ylabel(ylabel)
        sns.despine()

        stem = Path(csv_path).stem
        self._save_pdf(f"anchoring_correlation_{stem}_{self._get_timestamp()}")
        plt.close()
        return points, r

    def generate_runtime_plot(self, csv_paths, time_limit: float = None,
                              xlim: tuple = (10, 2000), clipped_at: float = 0.9):
        """
        Series 1-2. Wall time against the Boolean dimension of the benchmark,
        one marker per benchmark and per explainer, on log-log axes. One panel
        per task, so one call per optimization CSV.

        A cactus plot would have been the reflex, but nothing fails here: every
        explainer returns an explanation on every instance, so no curve would
        ever stop and the figure would only show sorted times. What carries the
        argument is the slope: the time of IHT does not grow with the dimension
        while every other method climbs by two to three orders of magnitude over
        the same range, and pooling the benchmarks would hide exactly that.

        Markers are not joined. Each benchmark is its own model and its own
        difficulty, so consecutive dimensions carry no trajectory; joining them
        turned ordinary scatter between neighbouring benchmarks into zigzags
        that read as instability. A least-squares line per explainer carries the
        trend instead, and its slope is the exponent of the power law in d --
        near zero for IHT, near one for the baselines.

        Beyond a hundred dimensions the time of the exact solver is the time
        limit, not the cost it would have paid. Fitting over those values would
        not merely bias the slope, it would bend it towards the horizontal of
        the limit, so the line would understate the very growth it is meant to
        show. Its fit therefore uses only the benchmarks it solved below
        `clipped_at` times the limit, and is extrapolated across the panel:
        where the dashed line leaves the frame is where the solver would have
        been had it been left to finish.

        xlim is shared by both panels so that a dimension sits at the same place
        in each; it also suppresses the minor decade ticks, which overlap on the
        narrower of the two ranges.
        """
        if isinstance(csv_paths, (str, Path)):
            csv_paths = [csv_paths]
        df = pd.concat([pd.read_csv(path) for path in csv_paths], ignore_index=True)

        df["dimension"] = df["dataset"].map(self.DIMENSIONS)
        missing = sorted(set(df.loc[df["dimension"].isna(), "dataset"]))
        if missing:
            raise ValueError(f"No dimension recorded for {missing}.")

        order = [e for e in ["MIP", "IHT", "LIME", "MAPLE"]
                 if e in set(df["explainer"])]

        plt.figure(figsize=(7, 5))
        if time_limit:
            plt.axhline(time_limit, color="0.55", linestyle=":", linewidth=1.0,
                        zorder=0)
            plt.annotate(rf"MIP time limit ({time_limit:g} s)",
                         xy=(0.99, time_limit), xycoords=("axes fraction", "data"),
                         ha="right", va="bottom", fontsize="x-small", color="0.4")

        for explainer in order:
            grouped = (df[df["explainer"] == explainer]
                       .groupby("dimension")["walltime"].mean().sort_index())
            _, marker = self.STYLES[explainer]
            plt.scatter(grouped.index, grouped.values, s=38,
                        color=self.COLORS[explainer], marker=marker,
                        edgecolor="white", linewidth=0.4, label=explainer, zorder=3)

            fitted = grouped
            if explainer == "MIP" and time_limit:
                fitted = grouped[grouped < clipped_at * time_limit]
            if len(fitted) < 2:
                continue

            # Least squares in log-log, accumulated from centred products: the
            # slope is the exponent of the power law, and nothing here goes
            # through a matrix product.
            lx = np.log10(fitted.index.to_numpy(dtype=float))
            ly = np.log10(fitted.to_numpy(dtype=float))
            cx, cy = lx - lx.mean(), ly - ly.mean()
            sxx = float((cx * cx).sum())
            if not sxx:
                continue
            slope = float((cx * cy).sum()) / sxx
            intercept = ly.mean() - slope * lx.mean()

            # Drawn across the whole panel rather than across the fitted points,
            # so that the trend of a clipped solver stays readable past the
            # dimensions where it still finished.
            span = np.log10(np.array(xlim if xlim else
                                     [grouped.index.min(), grouped.index.max()],
                                     dtype=float))
            plt.plot(10 ** span, 10 ** (slope * span + intercept),
                     color=self.COLORS[explainer], linestyle="--",
                     linewidth=1.0, alpha=0.55, zorder=2)

        plt.xscale("log")
        plt.yscale("log")
        if xlim:
            plt.xlim(*xlim)
        plt.xlabel(r"Dimension $d$")
        plt.ylabel(r"Wall time (s)")
        plt.legend(title="Explainer", loc="upper left", framealpha=1.0)
        sns.despine()

        self._save_pdf(f"runtime_vs_dimension_{self._get_timestamp()}")
        plt.close()

    def generate_sigma_sweep_plot(self, csv_path: str):
        """
        Series 7-8. Relevance against the concentration of the neighbourhood.
        The budget sweep shows the hierarchy holds at every sparsity level; this
        one answers the question it raises, namely whether it also holds as the
        neighbourhood widens from the reference instance to the uniform
        distribution.

        Relevance, not the ratio R / F. The ratio is what Lemma 1 constrains, so
        it looks like the natural quantity to sweep, but it degenerates in the
        local regime: as sigma grows nearly every draw equals the reference
        instance, the fidelity of an anchored explanation falls to zero, and the
        ratio is undefined -- the more evaluation samples, the more often this
        happens. Relevance has no such flaw, and the comparison between
        explainers is what the figure is for.

        Same grouped bars as the budget sweep: the exact solver and IHT reach
        the same relevance almost everywhere, and equal heights side by side
        read as agreement where two crossing lines would read as noise.
        """
        df = pd.read_csv(csv_path)

        def _plot(ds_df, ds_name):
            order = [e for e in ["MIP", "IHT", "LIME", "MAPLE", "CVX"]
                     if e in set(ds_df["explainer"])]
            sigmas = sorted(ds_df["sigma"].unique())
            positions = {s: i for i, s in enumerate(sigmas)}

            width = 0.8 / len(order)
            for index, explainer in enumerate(order):
                mean = (ds_df[ds_df["explainer"] == explainer]
                        .groupby("sigma")["relevance"].mean())
                offset = (index - (len(order) - 1) / 2) * width
                plt.bar(
                    [positions[s] + offset for s in mean.index], mean.values,
                    width=width * 0.92, label=explainer,
                    color=self.COLORS[explainer], edgecolor="white", linewidth=0.6,
                    zorder=3,
                )

            plt.xticks(range(len(sigmas)), [f"{s:g}" for s in sigmas])
            plt.xlabel(r"Concentration Level ($\sigma$)")
            plt.ylabel(r"Relevance Error ($\mathsf{R}$)")
            plt.legend(title="Explainer", loc="upper right", framealpha=1.0)
            plt.grid(axis="x", visible=False)

        self._plot_dataset_loop(df, "sigma_sweep", _plot)

    def generate_m_sweep_plot(self, csv_path: str):
        """
        Series 9-10. Relevance against the number of local samples used to build
        the empirical objective. The budget and concentration sweeps leave the
        practical question open: how many samples does an explanation need
        before it is worth using?

        Relevance again, not the generalization gap |F - F_hat|. The gap is what
        the sample complexity theorem bounds, but its decrease with m is a
        foregone conclusion, and it would break the reading of the other panels
        by switching metric. The theorem says that beyond some m the bound holds
        with high probability; what the figure shows is the empirical companion
        of that statement, namely that the sample size needed in practice is
        small.

        Relevance is estimated by Monte Carlo, so it has a noise floor of about
        1 / (2 sqrt(eval_samples)). A curve that flattens near that floor has
        stopped measuring the method.
        """
        df = pd.read_csv(csv_path)

        def _plot(ds_df, ds_name):
            order = [e for e in ["MIP", "IHT", "LIME", "MAPLE", "CVX"]
                     if e in set(ds_df["explainer"])]
            sizes = sorted(ds_df["m"].unique())
            positions = {m: i for i, m in enumerate(sizes)}

            width = 0.8 / len(order)
            for index, explainer in enumerate(order):
                mean = (ds_df[ds_df["explainer"] == explainer]
                        .groupby("m")["relevance"].mean())
                offset = (index - (len(order) - 1) / 2) * width
                plt.bar(
                    [positions[m] + offset for m in mean.index], mean.values,
                    width=width * 0.92, label=explainer,
                    color=self.COLORS[explainer], edgecolor="white", linewidth=0.6,
                    zorder=3,
                )

            plt.xticks(range(len(sizes)), [f"{m:g}" for m in sizes])
            plt.xlabel(r"Number of Samples ($m$)")
            plt.ylabel(r"Relevance Error ($\mathsf{R}$)")
            plt.legend(title="Explainer", loc="upper right", framealpha=1.0)
            plt.grid(axis="x", visible=False)

        self._plot_dataset_loop(df, "m_sweep", _plot)

    def generate_convergence_plot(self, csv_path: str):
        """
        Series 13-14. The ratio of empirical fidelities against sigma, which is
        the multiplicative form the end-to-end theorem takes. The horizontal at 1
        is the exact optimum, reachable only where the MIP proves optimality --
        which is why this series runs on small benchmarks.
        """
        df = pd.read_csv(csv_path)
        keys = ['dataset', 'sigma', 'run']
        iht = df[df['explainer'] == 'IHT'].set_index(keys)['emp_fidelity']
        mip = df[df['explainer'] == 'MIP'].set_index(keys)['emp_fidelity']

        ratio = (iht / mip.replace(0.0, np.nan)).rename('ratio').reset_index()
        ratio = ratio.dropna(subset=['ratio'])

        def _plot(ds_df, ds_name):
            sns.lineplot(
                data=ds_df, x='sigma', y='ratio', marker='s', linewidth=1.5,
                color="#3E65FE", errorbar=('ci', 95), label=r"IHT / MIP",
            )
            plt.axhline(1.0, color='gray', linestyle='--', linewidth=1.0, zorder=0,
                        label=r"Exact optimum")
            plt.xscale('symlog', linthresh=0.01)
            plt.xlim(left=-0.002)
            plt.xlabel(r"Concentration Level ($\sigma$)")
            plt.ylabel(r"$\widehat{\mathsf{F}}(\vec{w}_T) / \widehat{\mathsf{F}}(\hat{\vec{w}})$")
            plt.legend(loc="best")

        self._plot_dataset_loop(ratio, "convergence", _plot)

    def generate_time_sweep_plot(self, csv_path: str):
        """
        Series 11-12, appendix. Anytime behaviour of the exact solver on the two
        benchmarks where it never proves optimality within the limit.
        """
        df = pd.read_csv(csv_path)
        for column in ['expected_time', 'emp_fidelity']:
            df[column] = pd.to_numeric(df[column], errors='coerce')

        def _plot(ds_df, ds_name):
            sns.lineplot(
                data=ds_df, x='expected_time', y='emp_fidelity', hue='explainer',
                style='explainer', errorbar=('ci', 95), linewidth=1.5,
                **self._style_args(sorted(ds_df['explainer'].unique())),
            )
            plt.xlim(left=ds_df['expected_time'].min())

            proven = ds_df[(ds_df['explainer'] == 'MIP') & (ds_df['is_optimal'] == True)]
            if not proven.empty:
                avg_opt_time = proven.groupby('run')['expected_time'].min().mean()
                plt.axvline(x=avg_opt_time, color='gray', linestyle='--', alpha=0.7,
                            zorder=0, label='MIP proven optimal (avg)')

            plt.xlabel(r"Time Limit (s)")
            plt.ylabel(r"Empirical Fidelity Error ($\widehat{\mathsf{F}}$)")
            plt.legend(title="Explainer", loc="best")

        self._plot_dataset_loop(df, "time_sweep", _plot)