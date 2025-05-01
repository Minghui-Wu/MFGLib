from mfglib.env import Environment
from mfglib.alg import MFOMO,PriorDescent, FictitiousPlay,TwoScaleLearning,OnlineMirrorDescent
from mfglib.metrics import exploitability_score
import matplotlib.pyplot as plt
from mfglib.mean_field import mean_field
import torch

# instance = Environment.susceptible_infected(T=15)
env_instance = Environment.equilibrium_price()

# Run the MF-OMO algorithm with default hyperparameters and default tolerances and plot exploitability scores
solns, expls, runtimes = TwoScaleLearning().solve(env_instance, max_iter=500, verbose=True)
#solns, expls, runtimes = FictitiousPlay(update_initial=True).solve(instance, max_iter=3000, verbose=True)

plt.semilogy(runtimes, exploitability_score(env_instance, solns)) 
plt.grid(True)
plt.xlabel("Runtime (seconds)")
plt.ylabel("Exploitability")
plt.title("Rock Paper Scissors Environment - MFOMO Algorithm")
plt.show()