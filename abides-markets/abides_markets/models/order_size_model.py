import numpy as np
from scipy.stats import lognorm, norm


class OrderSizeModel:
    """Order size model using scipy distributions instead of pomegranate.
    
    This implementation replicates the behavior of the original pomegranate-based model
    using scipy distributions, avoiding the need for pomegranate which has compatibility
    issues with Python 3.12.
    """
    
    def __init__(self) -> None:
        # Distribution parameters from the original model
        self.distributions = [
            ('lognormal', {'s': 1.2, 'scale': np.exp(2.9)}),  # LogNormal(2.9, 1.2)
            ('normal', {'loc': 100.0, 'scale': 0.15}),
            ('normal', {'loc': 200.0, 'scale': 0.15}),
            ('normal', {'loc': 300.0, 'scale': 0.15}),
            ('normal', {'loc': 400.0, 'scale': 0.15}),
            ('normal', {'loc': 500.0, 'scale': 0.15}),
            ('normal', {'loc': 600.0, 'scale': 0.15}),
            ('normal', {'loc': 700.0, 'scale': 0.15}),
            ('normal', {'loc': 800.0, 'scale': 0.15}),
            ('normal', {'loc': 900.0, 'scale': 0.15}),
            ('normal', {'loc': 1000.0, 'scale': 0.15}),
        ]
        
        self.weights = np.array([
            0.2, 0.7, 0.06, 0.004, 0.0329, 0.001, 0.0006, 0.0004, 0.0005, 0.0003, 0.0003
        ])
        
        # Normalize weights to sum to 1
        self.weights = self.weights / self.weights.sum()

    def sample(self, random_state: np.random.RandomState) -> float:
        # Select a distribution based on weights
        dist_idx = random_state.choice(len(self.distributions), p=self.weights)
        dist_type, params = self.distributions[dist_idx]
        
        # Sample from the selected distribution
        if dist_type == 'lognormal':
            sample = lognorm.rvs(random_state=random_state, **params)
        else:  # normal
            sample = norm.rvs(random_state=random_state, **params)
        
        return round(sample)
