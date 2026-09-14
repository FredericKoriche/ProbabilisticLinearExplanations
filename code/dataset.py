import numpy as np
import pandas as pd
import warnings
import torch

from abc import ABC, abstractmethod
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from typing import List
from torch.utils.data import Dataset as TorchDataset
from utils import Console

class Feature:
    def __init__(self, index: int, name: str, size: int, distribution: np.ndarray):
        """
        Stores metadata and distribution of a single feature in the dataset:
        - index: Column index of this feature in X.
        - name: Original name of the feature.
        - size: Number of unique values this feature takes (will be 2 for {-1, 1}).
        - distribution: Empirical probability mass function (PMF).
        """
        self.index = index
        self.name = name
        self.size = size
        self.distribution = distribution

    def __str__(self):
        # Prevent mutating global numpy print options by formatting locally
        dist_str = np.array2string(self.distribution, precision=4, suppress_small=True)
        prefix = f"{self.index}: {self.name}"
        infix = f"({self.size} values)"
        suffix = f"Distribution: {dist_str}"
        return f"{prefix.ljust(30)} | {infix.ljust(12)} | {suffix}\n"


class Dataset(ABC, TorchDataset):
    def __init__(self, console: Console, id: int = -1, name: str = "", version: int = 1,
                 max_features: int = -1, n_bins: int = 4, verbose: bool = True, verbose_level: str = "light"):
        super().__init__()
        
        # Input parameters
        self.console = console
        self.id = id
        self.name = name
        self.version = version
        self.max_features = max_features
        self.n_bins = n_bins
        self.verbose = verbose
        self.verbose_level = verbose_level
        
        # Output data and metadata
        self.features: List[Feature] = []
        self.n_missing_values = 0
        self.n_raw_attributes = 0
        self.n_cat_attributes = 0
        self.n_num_attributes = 0        
        self.n_booleans = 0
        self.n_features = 0
        self.n_instances = 0

        # Preprocessing bookkeeping
        self.n_dropped_incomplete = 0
        self.n_dropped_high_cardinality = 0
        self.n_dropped_collapsed = 0
        
        # Pandas DataFrames for logic/processing
        self.raw_X = pd.DataFrame()
        self.raw_Y = pd.Series(dtype=object)
        self.X = pd.DataFrame()
        self.Y = pd.DataFrame()
        
        # PyTorch Tensors for the pipeline
        self.X_tensor = None
        self.Y_tensor = None

    def __len__(self):
        return len(self.X_tensor) if self.X_tensor is not None else 0

    def __getitem__(self, idx):
        return self.X_tensor[idx], self.Y_tensor[idx]

    def __str__(self):
        stats = {
            "ID": self.id,
            "Name": self.name,
            "Nb Missing Values": self.n_missing_values,
            "Nb Raw Attributes": self.n_raw_attributes,
            "Nb Categorical Attributes": self.n_cat_attributes,
            "Nb Numerical Attributes": self.n_num_attributes,
            "Nb Instances": self.n_instances,
            "Nb Processed Features": self.n_features,
            "Nb Domain Values (Bins)": self.n_bins
        }
        res = self.console.string("Dataset Statistics", endl=True)
        for k, v in stats.items():
            res += self.console.string(k, v, endl=True)
            
        if self.verbose_level in ["medium", "full"]:
            res += self.console.string("Feature Statistics", endl=True)
            for feature in self.features:
                res += str(feature)
        return res

    def _cleanData(self) -> None:
        """
        Drops rows with a missing target, then drops every attribute holding at
        least one missing value. No imputation takes place: after this step no
        missing value remains anywhere, which is what makes the rule statable in
        one sentence.
        """
        self.console.log("Cleaning dataset")
        missing_strings = ['?', '', '.', 'nan', 'na', 'none', 'null', 'None']
        self.raw_X = self.raw_X.replace(missing_strings, np.nan)

        # Target cleaning (drop rows where Y is NaN)
        valid_idx = self.raw_Y.dropna().index
        self.raw_X = self.raw_X.loc[valid_idx]
        self.raw_Y = self.raw_Y.loc[valid_idx]

        self.n_missing_values = int(self.raw_X.isnull().values.sum())

        if self.n_missing_values > 0:
            incomplete = self.raw_X.columns[self.raw_X.isnull().any()]
            self.n_dropped_incomplete = len(incomplete)
            self.console.log(
                "Dropping attributes with missing values",
                f"{self.n_dropped_incomplete} of {self.raw_X.shape[1]}",
                color="yellow",
            )
            self.raw_X = self.raw_X.drop(columns=incomplete)

    def _createTensors(self) -> None:
        # Detect GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.console.log(f"Creating PyTorch Tensors on {device}")
        
        # Push tensors
        self.X_tensor = torch.tensor(self.X.values, dtype=torch.float32, device=device)
        self.Y_tensor = torch.tensor(self.Y.values, dtype=torch.float32, device=device)

    def _discretizeData(self) -> None:
        """
        Encodes every attribute as binary indicators in {-1, +1}.

        An attribute taking at most n_bins distinct values is one-hot encoded,
        whatever its type: a numerical attribute with few values needs no
        discretization. A categorical attribute taking more values is reduced to
        its n_bins - 1 most frequent categories, all remaining values being
        grouped into a single residual category. A numerical attribute taking
        more values is cut into n_bins quantile bins, and is discarded when its
        quantile edges are too close for the discretizer to return n_bins bins,
        which happens for heavily skewed or zero-inflated attributes.

        Every retained attribute therefore contributes at most n_bins indicators,
        and a retained numerical attribute contributes exactly n_bins, so the
        final dimension can be recomputed from the attribute counts.
        """
        self.console.log("Discretizing and binarizing dataset to {-1, +1}")
        X = self.raw_X.loc[:, self.raw_X.nunique() > 1].copy()
        processed_cols = []

        cat_cols = X.select_dtypes(exclude=['number']).columns.tolist()
        num_cols = X.select_dtypes(include=['number']).columns.tolist()

        # A numerical attribute taking at most n_bins distinct values needs no
        # discretization: it is one-hot encoded like a categorical one.
        few_valued = [c for c in num_cols if X[c].nunique() <= self.n_bins]
        cat_cols += few_valued
        num_cols = [c for c in num_cols if c not in few_valued]

        # --- Categorical attributes ---------------------------------------------
        for col in cat_cols:
            series = X[col].astype(str)
            if series.nunique() > self.n_bins:
                top = series.value_counts().nlargest(self.n_bins - 1).index
                series = series.where(series.isin(top), other="Other")

            dummies = pd.get_dummies(series, prefix=col, dtype=int)
            dummies = dummies * 2 - 1  # Map {0, 1} -> {-1, 1}
            processed_cols.append(dummies)

        # --- Numerical attributes -----------------------------------------------
        # Keep the attributes whose n_bins quantile bins are all wide enough, so
        # that the discretizer returns exactly n_bins bins for each of them. The
        # 1e-8 threshold mirrors the criterion used by KBinsDiscretizer itself:
        # distinct edges are not enough, they must also be far enough apart.
        probs = np.linspace(0.0, 1.0, self.n_bins + 1)
        kept_num = []
        for col in num_cols:
            edges = np.quantile(X[col].to_numpy(dtype=float), probs)
            if np.all(np.diff(edges) > 1e-8):
                kept_num.append(col)

        self.n_dropped_collapsed = len(num_cols) - len(kept_num)
        if self.n_dropped_collapsed:
            self.console.log(
                "Dropping numerical attributes with collapsing quantiles",
                self.n_dropped_collapsed,
                color="yellow",
            )

        # Fit once, then discard the attributes for which the discretizer could
        # not return n_bins bins, and fit again on the survivors. This defers to
        # the discretizer's own criterion rather than trying to reproduce it.
        if kept_num:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                discretizer = KBinsDiscretizer(
                    n_bins=self.n_bins, encode='onehot-dense', strategy='quantile'
                )
                discretizer.fit(X[kept_num])

                collapsed = [
                    col for col, nb in zip(kept_num, discretizer.n_bins_)
                    if nb != self.n_bins
                ]
                if collapsed:
                    kept_num = [c for c in kept_num if c not in collapsed]
                    self.n_dropped_collapsed += len(collapsed)
                    self.console.log(
                        "Dropping numerical attributes rejected by the discretizer",
                        len(collapsed),
                        color="yellow",
                    )
                    discretizer = KBinsDiscretizer(
                        n_bins=self.n_bins, encode='onehot-dense', strategy='quantile'
                    )
                    discretizer.fit(X[kept_num])

            binned = discretizer.transform(X[kept_num]) * 2 - 1
            bin_cols = [
                f"{col}_bin_{j}" for col in kept_num for j in range(self.n_bins)
            ]
            df_bin_num = pd.DataFrame(binned, columns=bin_cols, index=X.index, dtype=int)
            processed_cols.append(df_bin_num)
            
        self.n_cat_attributes = len(cat_cols)
        self.n_num_attributes = len(kept_num)

        Z = pd.concat(processed_cols, axis=1).copy() if processed_cols else pd.DataFrame(index=X.index)
        self.X = Z.loc[:, Z.nunique() > 1]

        # With the filters above every indicator should already be non-constant;
        # a warning here signals that one of the selection rules is wrong.
        if self.X.shape[1] != Z.shape[1]:
            self.console.log(
                "Constant indicators removed after encoding",
                Z.shape[1] - self.X.shape[1],
                color="red",
            )

    def _downloadData(self) -> None:
        self.console.log("Loading dataset", self.name)
        fetch_kwargs = {"data_id": self.id} if self.id != -1 else {"name": self.name, "version": self.version}
        X, y = fetch_openml(**fetch_kwargs, parser='auto', as_frame=True, return_X_y=True)
        self.raw_X = X
        self.raw_Y = y
        self.n_raw_attributes = self.raw_X.shape[1]

    @abstractmethod
    def _formatTarget(self) -> None:
        """To be implemented by subclasses to format Y into the desired domain."""
        pass

    @abstractmethod
    def _get_feature_selector_model(self):
        """Returns the appropriate scikit-learn model for feature selection."""
        pass

    def _selectFeatures(self) -> None:
        if self.X.shape[1] <= self.max_features or self.max_features <= 0:
            return
        
        self.console.log(f"Selecting top {self.max_features} features")
        selector_model = self._get_feature_selector_model()
        selector = SelectFromModel(selector_model, max_features=self.max_features, threshold=-np.inf)
        
        selector.fit(self.X, self.Y.values.ravel())
        selected_mask = selector.get_support()
        
        self.console.log(f"Dimension reduced", f"{self.X.shape[1]} -> {len(selected_mask[selected_mask])}")
        self.X = self.X.loc[:, selected_mask]

    def _setDistributions(self) -> None:
        self.console.log("Setting feature distributions for {-1, +1} space")
        self.features = [] 
        
        for i, col_name in enumerate(self.X.columns):
            series = self.X[col_name]
            size = 2 # Absence (-1) vs Presence (+1)
            
            prob_pos = (series == 1).mean()
            prob_neg = (series == -1).mean()
            
            # Index 0 corresponds to -1, Index 1 corresponds to +1
            distribution = np.array([prob_neg, prob_pos])
            distribution = (distribution + 1e-12)
            distribution /= distribution.sum()
            
            self.features.append(Feature(index=i, name=col_name, size=size, distribution=distribution)) 
            
        self.n_booleans = sum(feature.size for feature in self.features)
        self.n_features = len(self.features)
        self.n_instances = self.X.shape[0]

    def build(self) -> str:
        """Runs the full data loading and preprocessing pipeline."""
        self.console.log("Setup dataset", f"(ID: {self.id})")
        self._downloadData()
        self._cleanData()
        self._discretizeData()
        self._formatTarget()
        # Feature selection is kept available but is not part of the reported
        # pipeline: it would introduce a second model (and a second seed) into
        # the construction of the data. Uncomment to reduce the dimension, e.g.
        # to inspect explanations over a small number of candidate features.
        # self._selectFeatures()
        self._setDistributions()
        self._createTensors()

        # Log the exact number of binary features generated
        self.console.log("Final Binary Feature Dimension (d)", self.X.shape[1])
        return self.name

class ClassificationDataset(Dataset):
    OPENML = {
        13: "Breast Cancer",
        15: "Breast Cancer (WI)",
        24: "Mushroom",
        27: "Colic",
        29: "Credit Approval",
        31: "Credit-G",
        37: "Pima Indians Diabetes",
        40: "Sonar",
        44: "Spambase",
        50: "Tic-Tac-Toe",
        55: "Hepatitis",
        56: "Vote",
        59: "Ionosphere",
        179: "Adult",
        335: "Monks 3",
        1116: "Musk",
        1461: "Bank Marketing",
        1480: "Indian Liver Patients",
        1486: "Nomao",
        23512: "Higgs",
        40536: "Speed Dating",
        40683: "Postoperative", 
        40981: "Australian",
        42192: "COMPAS", 
        43672: "Heart Disease", 
        45578: "California Housing"
    }

    def __init__(self, console: Console, id: int = -1, name: str = "", version: int = 1,
                 max_features: int = -1, n_bins: int = 4, verbose: bool = True, verbose_level: str = "light"):
        super().__init__(console, id, name, version, max_features, n_bins, verbose, verbose_level)
        if not self.name and id != -1:
            self.name = self.OPENML[id]
        self.majority_class = None

    def _formatTarget(self) -> None:
        """Encode labels to binary form in {-1, +1}."""
        self.console.log("Encoding labels to binary form {-1, +1}")
        counts = self.raw_Y.value_counts()
        self.majority_class = counts.idxmax()
        
        bin_labels = np.where(self.raw_Y == self.majority_class, 1, -1)
        self.Y = pd.DataFrame(bin_labels, columns=["Label"], index=self.raw_X.index)

    def _get_feature_selector_model(self):
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    def __str__(self):
        res = super().__str__()
        res += self.console.string("Majority Class", self.majority_class, endl=True)
        return res


class RegressionDataset(Dataset):
    OPENML = {
        8: "Liver Disorders",
        9: "Automobile",
        191: "Wisconsin",
        194: "Cleveland",
        204: "Cholesterol",
        206: "Triazines",
        223: "Stock Prices",
        230: "Machine CPU",
        287: "Wine Quality",
        511: "Plasma Retinol",
        542: "Pollution",
        566: "Meta",
        1089: "US Crime",
        41021: "Moneyball",
        42225: "Diamonds",
        42352: "Student Performance",
        42363: "Forest Fires",
        42372: "Auto MPG",
        42724: "Online News Popularity",
        42726: "Abalone",
        44019: "House Sales",
        44024: "California Housing",
        44032: "Fifa",
        44042: "Black Friday",
        44133: "Pol",
        44134: "Elevator",
        44137: "Ailerons",
        44139: "House 16H",
        44141: "Brazilian Houses",
        44142: "Bike Sharing Demand",
        44143: "NYC Taxi Green",
        44145: "Sulfur",
        44146: "Medical Charges",  # Use 10 bins
        44957: "Airfoil Self Noise",
        44958: "Auction Verification",
        44959: "Concrete Compressive Strength",
        44962: "Forest Fires",
        44963: "Physicochemical Protein",
        44965: "Geographical Origin of Music",
        44966: "Solar Flare",
        44969: "Naval Propulsion Plant",
        44970: "Fish Toxicity",
        44973: "Grid Stability",
        44975: "Wave Energy",
        44983: "Miami Housing",
        44984: "CPS88 Wages",
        44987: "Socmob",
        44989: "Kings County",
        44994: "Cars",
        46132: "NCI 60 Thioguanine",
        46134: "Acute Myeloid Leukemia",
        46139: "Cancer Drug Response Methylation",
        46283: "Appliances Energy Prediction",
        46286: "Communities and Crime",
        46328: "Seoul Bike Sharing",
        46337: "Conso RTE"
    }    
    
    def __init__(self, console: Console, id: int = -1, name: str = "", version: int = 1,
                 max_features: int = -1, n_bins: int = 4, verbose: bool = True, verbose_level: str = "light"):
        super().__init__(console, id, name, version, max_features, n_bins, verbose, verbose_level)
        if not self.name and id != -1:
            self.name = self.OPENML[id]

    def _downloadData(self) -> None:
        """Override to ensure raw_Y is converted to numeric for regression."""
        super()._downloadData()
        self.raw_Y = pd.to_numeric(self.raw_Y, errors='coerce')

    def _formatTarget(self) -> None:
        """Scales regression labels to [-1, +1] range."""
        self.console.log("Scaling labels to [-1, +1]")
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_y = scaler.fit_transform(self.raw_Y.to_numpy().reshape(-1, 1))
        self.Y = pd.DataFrame(scaled_y, columns=["Label"], index=self.raw_X.index)

    def _get_feature_selector_model(self):
        return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)