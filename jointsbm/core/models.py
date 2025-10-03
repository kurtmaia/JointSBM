from dataclasses import dataclass

import numpy as np


@dataclass
class SBMResults:
    memberships: dict
    theta: np.ndarray

    def _labels(self):
        return {
            graph_name: memberships.argmax(axis=1)
            for graph_name, memberships in self.memberships.items()
        }
