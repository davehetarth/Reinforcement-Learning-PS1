import numpy as np
import gymnasium as gym

def value_iteration(P, gamma=0.99, theta=1e-4):
    num_states = len(P)
    num_actions = len(P[0])
    
    V = np.zeros(num_states)
    iteration_count = 0
    num_backups = 0
    
    optimal_policy_found_at = None
    max_norm_at_k_star = None
    
    target_optimal_policy = np.array([0, 3, 3, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0])
    
    while True:
        delta = 0
        V_new = np.zeros(num_states)
        iteration_count += 1
        
        for s in range(num_states):
            q_values = np.zeros(num_actions)
            for a in range(num_actions):
                num_backups += 1 
                for prob, next_state, reward, terminated in P[s][a]:
                    V_new_target = 0 if terminated else V[next_state]
                    q_values[a] += prob * (reward + gamma * V_new_target)
            
            V_new[s] = np.max(q_values)
            delta = max(delta, abs(V_new[s] - V[s]))
            
        current_policy = np.zeros(num_states, dtype=int)
        for s in range(num_states):
            q_vals = np.zeros(num_actions)
            for a in range(num_actions):
                for prob, next_state, reward, terminated in P[s][a]:
                    V_target = 0 if terminated else V_new[next_state]
                    q_vals[a] += prob * (reward + gamma * V_target)
            current_policy[s] = np.argmax(q_vals)
            
        if optimal_policy_found_at is None and np.array_equal(current_policy, target_optimal_policy):
            optimal_policy_found_at = iteration_count
            max_norm_at_k_star = delta

        V = V_new
        if delta < (theta * (1 - gamma) / gamma):
            break
            
    return V, current_policy, iteration_count, num_backups, optimal_policy_found_at, max_norm_at_k_star

def render_grids(V, policy, iteration_count):
    print(f"=== PART (a) REPORT ===")
    print(f"Iteration count to converge: {iteration_count}\n")
    
    print("--- Full V* Table (4x4 Grid) ---")
    v_grid = V.reshape(4, 4)
    for row in v_grid:
        print(" | ".join(f"{val:6.4f}" for val in row))
    print("\n")
    
    print("--- Policy pi* Rendered as Arrows ---")
    arrow_map = {0: "◄", 1: "▼", 2: "►", 3: "▲"}
    holes = {5, 7, 11, 12}
    goal = 15
    
    policy_grid = policy.reshape(4, 4)
    for r in range(4):
        row_str = []
        for c in range(4):
            idx = r * 4 + c
            if idx in holes:
                row_str.append(" █ ")
            elif idx == goal:
                row_str.append(" 🏁 ")
            else:
                row_str.append(f" {arrow_map[policy_grid[r, c]]} ")
        print("|".join(row_str))
    print("=" * 24 + "\n")

if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=True)
    V_vi, p_vi, it_vi, _, k_star, norm_k = value_iteration(env.unwrapped.P, gamma=0.99)
    render_grids(V_vi, p_vi, it_vi)
    
    print(f"=== PART (d) POLICY EMERGENCE ===")
    print(f"First iteration k* where policy equals pi*: {k_star}")
    print(f"||Vk* - Vk*-1||_inf value at that point: {norm_k:7.5f}")
    print("=" * 33)