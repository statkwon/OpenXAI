import torch
from iee import Explainer
from ...api import BaseExplainer


class IEE(BaseExplainer):
    def __init__(self, model, X=None, mode="classifier", grid_resolution=None):
        super().__init__(model)
        self._explainer = Explainer(model, X, mode, grid_resolution)

    def get_explanations(self, x: torch.FloatTensor, label=None) -> torch.FloatTensor:
        self.model.eval()
        iee_values = torch.FloatTensor(self._explainer(x))
        if iee_values.size()[-1] == 2:
            iee_values = iee_values[:, :, 1]
        return iee_values
