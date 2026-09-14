import torch
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from functools import partial
from sklearn.base import clone
from typing import Union

from sklearn.ensemble import (
    AdaBoostClassifier, AdaBoostRegressor, 
    GradientBoostingClassifier, GradientBoostingRegressor, 
    RandomForestClassifier, RandomForestRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from utils import Console


class BlackBox(ABC):
    # Will be overridden by child classes
    MODELS = {} 

    def __init__(self, *, name: str, seed: int = 48, verbose: bool = True, **kwargs):
        self.console = Console(verbose=verbose)

        if name not in self.MODELS:
            available = ", ".join(self.MODELS.keys())
            raise ValueError(f"{self.__class__.__name__} '{name}' not found. Available: {available}")

        self.name = name
        self.seed = seed

        model_template = self.MODELS[name]
        if isinstance(model_template, (type, partial)):
            self.model = model_template(**kwargs)
        else:
            self.model = clone(model_template)
            if kwargs:
                self.model.set_params(**kwargs)

        # Fix the seed on every estimator that exposes one, unless the caller
        # explicitly provided its own.
        if "random_state" in self.model.get_params() and "random_state" not in kwargs:
            self.model.set_params(random_state=self.seed)

        self.error = float("inf")
        self.dimension = 0
    
    def __str__(self):
        output = self.console.string(f"{self.__class__.__name__} Information", endl=True)
        output += self.console.string("Name", self.name, endl=True)
        output += self.console.string("Input Dimension", self.dimension, endl=True)
        output += self.console.string("Training Error", f"{self.error:.4f}", endl=True)
        return output

    def _calculate_error(self, Y: np.ndarray, P: np.ndarray) -> float:
        """
        Computes the universal normalized quadratic loss: 1/4 * (Y - P)^2.
        For {-1, +1} classification, this exactly coincides with the zero-one loss.
        For [-1, +1] regression, this acts as a bounded, scaled MSE.
        """
        return float(np.mean((Y - P) ** 2) / 4.0)

    def train(self, *, instances: pd.DataFrame, labels: pd.DataFrame) -> tuple:
        self.console.log(f"Training {self.name}")
        X = instances.to_numpy()
        Y = labels.to_numpy().ravel()
        
        self.model.fit(X, Y)
        self.dimension = X.shape[1]
        
        P = self.model.predict(X)
        self.error = self._calculate_error(Y, P)
        return self.dimension, self.error

    def predict(self, X: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Universal prediction bridge for PyTorch and NumPy"""
        if isinstance(X, np.ndarray):
            X_tensor = torch.tensor(X, dtype=torch.float32)
        else:
            X_tensor = X
            
        original_device = X_tensor.device
        X_np = X_tensor.detach().cpu().numpy()
            
        preds_np = self.model.predict(X_np)
        
        # Push back to original device, ensuring it doesn't hook into autograd
        with torch.no_grad():
            return torch.tensor(preds_np, dtype=torch.float32, device=original_device)


class Classifier(BlackBox):
    MODELS = {
        "AdaBoost": AdaBoostClassifier,
        "Decision Tree": DecisionTreeClassifier,
        "Random Forest": RandomForestClassifier,
        "Gradient Boosting": GradientBoostingClassifier,
        "Linear SVM": LinearSVC(dual="auto"), 
        "Neural Network": MLPClassifier,
        "NN Linear": partial(MLPClassifier, activation='identity'),
        "NN Smooth": partial(MLPClassifier, activation='tanh', hidden_layer_sizes=(100, 50)),
        "NN Sharp": partial(MLPClassifier, activation='relu', hidden_layer_sizes=(100, 50)),
    }


class Regressor(BlackBox):
    MODELS = {
        "AdaBoost": AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=3), n_estimators=100, random_state=48),
        "Bayesian Ridge": BayesianRidge(),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, max_leaf_nodes=4),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=48),
        "Lasso": Lasso(alpha=0.1),
        "Linear": LinearRegression(),
        "Neural Network": MLPRegressor(hidden_layer_sizes=(150, 100, 50), max_iter=500, activation="relu", solver="adam"),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=48),
        "Ridge": Ridge(alpha=1.0),
        "SVR": SVR(kernel="linear", C=1, epsilon=0.1),
    }