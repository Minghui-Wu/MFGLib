from __future__ import annotations

import torch

from mfglib.env import Environment

class TransitionFn:
    def __init__(self, M: int) -> None:
        self.p = torch.zeros(M, M, M)
        for s in range(M):
            for a in range(M):
                # If action a is chosen, the next state is a with probability 1.
                self.p[a, s, a] = 1.0

    def __call__(self, env: Environment, t: int, L_t: torch.Tensor) -> torch.Tensor:
        return self.p
    
class RewardFn:
    def __init__(
        self,
        inertia: float = 0,
        alpha: float = 10.0,
        beta: float = 5.0,
        gamma: float = 15.0,
        r: float = 2.0,
        inflow: float = 6000.0,
        Cap: float = 3000.0,
        dt: float = 1.0,
    ) -> None:
        self.inertia = inertia
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.r = r
        self.inflow = inflow
        self.Cap = Cap
        self.dt = dt

    def T_vectorized(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Compute T(s, mu) for each state s in a vectorized way.
        mu is a tensor of shape (M,).
        """
        M = mu.shape[0]
        indices = torch.arange(M, dtype=torch.float32)
        # Compute the temporary array: inflow * mu / Cap.
        temp = self.inflow * mu / self.Cap
        # Cumulative sum adjusted by the time step.
        sum_record = torch.cumsum(temp, dim=0) - self.dt * indices
        # Compute the running minimum.
        min_record, _ = torch.cummin(sum_record, dim=0)
        return sum_record - min_record  # shape: (M,)

    def f_vectorized(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Compute f(s, mu) for each state s.
        Returns a tensor of shape (M,).
        """
        M = mu.shape[0]
        s_vals = torch.arange(M, dtype=torch.float32)
        T_vals = self.T_vectorized(mu)  # shape: (M,)
        # Compute the threshold: r - s*dt - T(s, mu)
        threshold = self.r - s_vals * self.dt - T_vals
        # Use torch.where for the piecewise definition.
        f_vals = self.alpha * T_vals + torch.where(
            threshold > 0,
            self.beta * threshold,
            self.gamma * (-threshold)
        )
        return f_vals  # shape: (M,)

    def d_vectorized(self, M: int) -> torch.Tensor:
        """
        Compute d(s, a) for all state-action pairs.
        Returns a tensor of shape (M, M) where entry (s, a) = inertia * |s - a| * dt.
        """
        s_vals = torch.arange(M, dtype=torch.float32).unsqueeze(1)  # shape: (M, 1)
        a_vals = torch.arange(M, dtype=torch.float32).unsqueeze(0)  # shape: (1, M)
        return self.inertia * torch.abs(s_vals - a_vals) * self.dt  # shape: (M, M)

    def __call__(self, env, t: int, L_t: torch.Tensor) -> torch.Tensor:
        """
        Compute the reward matrix R of shape (M, M) for each (state s, action a):
            R[s, a] = f(s, mu) + d(s, a)
        L_t is expected to be a tensor of shape env.S (i.e. (M,)) representing the current state distribution.
        """
        l_s = len(env.S)
        mu_t = L_t.flatten(start_dim=l_s).sum(-1)
        M = mu_t.shape[0]
        # Compute f(s, mu) for each state.
        f_vals = self.f_vectorized(mu_t)  # shape: (M,)
        # Compute d(s, a) for all state-action pairs.
        d_vals = self.d_vectorized(M)   # shape: (M, M)
        # Broadcast f_vals over actions and sum.
        R = -f_vals.unsqueeze(1) - d_vals  # shape: (M, M)
        return R