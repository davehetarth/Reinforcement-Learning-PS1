import time
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from vi import value_iteration
from pi import policy_iteration

env = gym.make("FrozenLake-v1", is_slippery=True)
P_model = env.unwrapped.P

gammas = [0.5, 0.9, 0.99, 0.999]
vi_iters, pi_iters = [], []

print(f"=== PART (c) EMPIRICAL COMPARISON ===")
print(f"{'Gamma':<6} | {'Algo':<4} | {'Iterations':<10} | {'Time (s)':<10} | {'Est. Backups':<12}")
print("-" * 55)

for g in gammas:
    start = time.time()
    V_vi, p_vi, it_vi, back_vi, _, _ = value_iteration(P_model, gamma=g)
    time_vi = time.time() - start
    vi_iters.append(it_vi)
    print(f"{g:<6} | VI   | {it_vi:<10} | {time_vi:.5f}  | {back_vi:<12}")
    
    start = time.time()
    V_pi, p_pi, it_pi, back_pi = policy_iteration(P_model, gamma=g)
    time_pi = time.time() - start
    pi_iters.append(it_pi)
    print(f"{g:<6} | PI   | {it_pi:<10} | {time_pi:.5f}  | {back_pi:<12}")
    print("-" * 55)

plt.figure(figsize=(9, 5.5))
plt.plot(gammas, vi_iters, marker='o', color='royalblue', linewidth=2, label='Value Iteration (VI)')
plt.plot(gammas, pi_iters, marker='s', color='darkorange', linewidth=2, label='Policy Iteration (PI)')
plt.xscale('log')
plt.xticks(gammas, labels=[str(g) for g in gammas])
plt.xlabel(r'Discount Factor Horizon ($\gamma$)', fontsize=11)
plt.ylabel('Outer Computational Steps to Convergence', fontsize=11)
plt.title('Algorithmic Execution Horizon Scaling Over Lambda Configurations', fontsize=12, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()