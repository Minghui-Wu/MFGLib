from __future__ import annotations

import time
from typing import Literal, cast

import optuna
import torch

from mfglib.alg.abc import Algorithm
from mfglib.alg.q_fn import QFn
from mfglib.alg.utils import (
    _ensure_free_tensor,
    _print_fancy_header,
    _print_fancy_table_row,
    _print_solve_complete,
    _trigger_early_stopping,
)
from mfglib.env import Environment
from mfglib.mean_field import mean_field
from mfglib.metrics import exploitability_score


class TwoScaleLearning(Algorithm):
    """Online Mirror Descent algorithm.

    Notes
    -----
    See [#omd1]_ for algorithm details.

    .. [#omd1] Perolat, Julien, et al. "Scaling up mean field games with online mirror
        descent." arXiv preprint arXiv:2103.00623 (2021). https://arxiv.org/abs/2103.00623
    """

    def __init__(self, Q_speed: float = 0.6, mu_speed: float = 1.0) -> None:
        """Online Mirror Descent algorithm.

        Attributes
        ----------
        alpha
            Learning rate hyperparameter.
        """
        self.Q_speed = Q_speed
        self.mu_speed = mu_speed

    def __str__(self) -> str:
        """Represent algorithm instance and associated parameters with a string."""
        return f"OnlineMirrorDescent(Q_speed={self.Q_speed}, mu_speed={self.mu_speed})"

    def cal_diff(self,mu0, mu1):
        if len(mu0.shape) == 1:
            return torch.sum(torch.abs(mu0-mu1))
        else:
            return torch.max(torch.sum(abs(mu0-mu1), axis=1))

    def solve(
        self,
        env_instance: Environment,
        *,
        pi: Literal["uniform"] | torch.Tensor = "uniform",
        max_iter: int = 100,
        atol: float | None = 1e-3,
        rtol: float | None = 1e-3,
        verbose: bool = False,
    ) -> tuple[list[torch.Tensor], list[float], list[float]]:
        """Run the algorithm and solve for a Nash-Equilibrium policy.

        Args
        ----
        env_instance
            An instance of a specific environment.
        pi
            A numpy array of size (T+1,)+S+A representing the initial policy.
            If 'uniform', the initial policy will be the uniform distribution.
        max_iter
            Maximum number of iterations to run.
        atol
            Absolute tolerance criteria for early stopping.
        rtol
            Relative tolerance criteria for early stopping.
        verbose
            Print convergence information during iteration.
        """
        T = env_instance.T
        S = env_instance.S
        A = env_instance.A

        y = torch.zeros((T + 1,) + S + A)

        # Auxiliary functions
        soft_max = torch.nn.Softmax(dim=-1)

        # Auxiliary variables
        l_s = len(S)

        pi = _ensure_free_tensor(pi, env_instance)
        L = mean_field(env_instance, pi)
        mu = torch.sum(L, axis=2)

        solutions_pi = [pi]
        solutions_IC = [mu[0]]
        argmin = 0
        scores_exp = [exploitability_score(env_instance, pi)]
        scores_diff = [self.cal_diff(mu[0], mu[-1])]

        runtimes = [0.0]

        if verbose:
            _print_fancy_header(
                alg_instance=self,
                env_instance=env_instance,
                max_iter=max_iter,
                atol=atol,
                rtol=rtol,
            )
            _print_fancy_table_row(
                n=0,
                score_n=scores_exp[0],
                score_0=scores_exp[0],
                argmin=argmin,
                runtime_n=runtimes[0],
            )

        if _trigger_early_stopping(scores_exp[0], scores_exp[0], atol, rtol):
            if verbose:
                _print_solve_complete(seconds_elapsed=runtimes[0])
            return solutions_pi, solutions_IC, scores_exp, scores_diff, runtimes

        t = time.time()
        for n in range(1, max_iter + 1):
            # Mean-field corresponding to the policy
            L = mean_field(env_instance, pi)
            mu = torch.sum(L, axis=2)

            # Q-function corresponding to the policy and mean-field
            Q = QFn(env_instance, L, verify_integrity=False).for_policy(pi)

            # Update y and pi
            weight_Q = 1 / (n+1) **  self.Q_speed
            y += weight_Q * Q

            weight_mu = 1 / (n+1) ** self.mu_speed
            new_mu0 = (1-weight_mu) * mu[0] + weight_mu * mu[-1]
            new_mu0 /= torch.sum(new_mu0)
            env_instance.update_initial_distribution(new_mu0)

            pi = cast(
                torch.Tensor,
                soft_max(y.flatten(start_dim=1 + l_s)).reshape((T + 1,) + S + A),
            )

            L = mean_field(env_instance, pi)
            mu = torch.sum(L, axis=2)
            solutions_pi.append(pi.clone().detach())
            solutions_IC.append(mu[0].clone().detach())
            scores_exp.append(exploitability_score(env_instance, pi))
            scores_diff.append(self.cal_diff(mu[0], mu[-1]))

            if scores_exp[n] < scores_exp[argmin]:
                argmin = n
            runtimes.append(time.time() - t)

            if verbose:
                _print_fancy_table_row(
                    n=n,
                    score_n=scores_exp[n],
                    score_0=scores_exp[0],
                    argmin=argmin,
                    runtime_n=runtimes[n],
                )

            if _trigger_early_stopping(scores_exp[0], scores_exp[n], atol, rtol):
                if verbose:
                    _print_solve_complete(seconds_elapsed=runtimes[n])
                return solutions_pi, solutions_IC, scores_exp, scores_diff, runtimes

        if verbose:
            _print_solve_complete(seconds_elapsed=time.time() - t)

        return solutions_pi, solutions_IC, scores_exp, scores_diff, runtimes

    @classmethod
    def _init_tuner_instance(cls, trial: optuna.Trial) -> TwoScaleLearning:
        return TwoScaleLearning(
            alpha=trial.suggest_float("Q_speed", "mu_speed", 1e-5, 1e5, log=True),
        )
