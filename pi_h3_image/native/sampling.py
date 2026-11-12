"""Flow sampling on the sigma grid of a MiniMax-H3-Image checkpoint (ComfyUI comfy/ldm/minimax/sampling.py).

The checkpoint's output heads are distilled on its own grid (pdd_sigmas). sigma() is linear in the grid index, so
Forge's stock schedules at pdd_steps steps land on the grid the heads were made for; the flow shift is unused.

One step past ComfyUI: Simple reads the 1000-entry sigma table at whole timesteps, which puts a 6-step schedule up to
0.001 off the grid. Positions that close to a grid point are snapped onto it, so 6 steps (and 3, 2 or 1) give the
checkpoint's sigmas exactly, as MiniMax's own pipeline uses them.
"""

import torch
from backend.modules.k_prediction import PredictionDiscreteFlow

# in grid intervals: a 6-step Simple schedule is at most 0.004 off a grid point
SNAP = 0.01


class PredictionH3Image(PredictionDiscreteFlow):
    def __init__(self, model_config, grid):
        # the grid decreases from 1 to 0; kept ascending so index 0 is sigma 0. Set before the base class builds its
        # sigma table through sigma()
        object.__setattr__(self, "_grid", tuple(float(s) for s in reversed(tuple(grid))))
        # the checkpoint's own grid; _grid is this grid moved by the [H3] Resolution shift
        object.__setattr__(self, "_stock_grid", self._grid)
        object.__setattr__(self, "odds_factor", 1.0)
        super().__init__(model_config)

    def set_odds_factor(self, factor: float) -> None:
        """Move the grid by the [H3] Resolution shift (resolution_shift.odds_factor) and rebuild the sigma table, so
        Forge's schedules land on the moved grid as they land on the stock one."""
        from ..resolution_shift import shift_sigma
        factor = float(factor)
        object.__setattr__(self, "odds_factor", factor)
        object.__setattr__(self, "_grid", tuple(shift_sigma(s, factor) for s in self._stock_grid))
        self.set_parameters()

    @property
    def grid(self) -> list[float]:
        """The grid the heads are picked on, decreasing from 1 to 0 (the stock grid moved by the odds factor)."""
        return list(reversed(self._grid))

    def _sigma(self, timestep, snap):
        timestep = torch.as_tensor(timestep, dtype=torch.float32)
        grid = torch.tensor(self._grid, dtype=torch.float32, device=timestep.device)
        n = grid.shape[0] - 1
        pos = (timestep / self.multiplier).clamp(0.0, 1.0) * n
        if snap:
            nearest = pos.round()
            # never onto sigma 0: the table's first entry is the model's sigma_min, and a schedule that reaches a
            # zero sigma before its last step (Normal) or takes its logarithm divides by zero
            pos = torch.where(((pos - nearest).abs() < SNAP) & (nearest > 0), nearest, pos)
        i = pos.floor().long().clamp(max=n - 1)
        return torch.lerp(grid[i], grid[i + 1], pos - i)

    def sigma(self, timestep):
        return self._sigma(timestep, snap=True)

    def percent_to_sigma(self, percent):
        # not snapped: callers ask for a point just inside an end on purpose, such as ER SDE moving a first sigma of 1
        # (an infinite log-SNR) to percent_to_sigma(1e-4); snapped, that came back as 1.0 and the sampler made NaN
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return float(self._sigma(torch.tensor((1.0 - percent) * self.multiplier), snap=False))
