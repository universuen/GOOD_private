from __future__ import annotations

import pickle
from pathlib import Path

from torch import Tensor
import numpy as np


class History:
    def __init__(
            self,
            name: str,
            config_name: str,
    ):
        self.name = name
        self.values = []
        results_dir = Path(__file__).absolute().parent / 'results'
        results_dir.mkdir(exist_ok=True)
        self.path = results_dir / config_name / f'{name}.history'

    def __getitem__(self, item: int):
        return self.values[item]

    @property
    def last_one(self):
        return self[-1]

    @property
    def avg_value(self):
        return sum(self.values) / len(self.values)

    @property
    def max_value(self):
        return max(self.values)

    @property
    def std_deviation(self):
        return np.std(self.values)

    def append(self, value: float | Tensor):
        if type(value) is Tensor:
            value = value.item()
        self.values.append(value)

    def save(self, path: Path | str = None):
        path = self.path if path is None else path
        with open(path, 'wb') as f:
            pickle.dump(
                {
                    'name': self.name,
                    'values': self.values,
                },
                f,
            )

    def load(self, path: Path | str = None):
        path = self.path if path is None else path
        with open(path, 'rb') as f:
            records = pickle.load(f)
            self.name = records['name']
            self.values = records['values']

    def __repr__(self):
        return f'<History | {self.name} | {len(self.values)} values | {self.avg_value}>'
