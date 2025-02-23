from mfglib.env import Environment
from mfglib.alg import MFOMO,PriorDescent, FictitiousPlay
from mfglib.metrics import exploitability_score
import matplotlib.pyplot as plt
from mfglib.mean_field import mean_field
import torch

# instance = Environment.susceptible_infected(T=15)
instance = Environment.beach_bar(n=10, bar_loc=2, T=15)

# Run the MF-OMO algorithm with default hyperparameters and default tolerances and plot exploitability scores
solns, expls, runtimes = PriorDescent(eta=1, n_inner=1, update_initial=False).solve(instance, max_iter=500, verbose=True)
#solns, expls, runtimes = FictitiousPlay(update_initial=True).solve(instance, max_iter=3000, verbose=True)

plt.semilogy(runtimes, exploitability_score(instance, solns)) 
plt.grid(True)
plt.xlabel("Runtime (seconds)")
plt.ylabel("Exploitability")
plt.title("Rock Paper Scissors Environment - MFOMO Algorithm")
plt.show()