import torch
from utils import Console

class Sampler:
    def __init__(self, console: Console, device: str = None):
        r"""
        A PyTorch-accelerated sampler for the distribution:
        D(z) \propto exp(-sigma/2 ||z - x||_1)
        in the {-1, +1}^d instance space.
        """
        self.console = console
        
        # Consistent hardware support (including Apple Silicon)
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        self.console.log("Initializing L1-Norm Parameterized Sampler", f"Device: {self.device}")

    def sample(self, x: torch.Tensor, sigma: float, n_samples: int) -> torch.Tensor:
        x = x.to(self.device)
        if x.dim() == 1: 
            x = x.unsqueeze(0)
        if x.shape[0] != 1: 
            raise ValueError(f"Expected x to be a single instance, got batch size {x.shape[0]}")
            
        # P(keep) = 1 / (1 + exp(-sigma))
        p_keep = torch.sigmoid(torch.tensor(float(sigma), device=self.device))
        
        # Create expanded tensor and flip bits based on random uniform threshold
        Z = x.expand(n_samples, x.shape[1]).clone()
        flip_mask = torch.rand(n_samples, x.shape[1], device=self.device) > p_keep
        Z[flip_mask] *= -1
        
        return Z