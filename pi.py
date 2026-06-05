import numpy as np
import gymnasium as gym

def policy_iteration(P, gamma=0.99):
    num_states = len(P)
    num_actions = len(P[0])
    
    policy = np.zeros(num_states, dtype=int)
    iteration_count = 0
    total_backups = 0
    
    while True:
        iteration_count += 1
        
        A_mat = np.eye(num_states)
        R_pi = np.zeros(num_states)
        
        for s in range(num_states):
            chosen_a = policy[s]
            for prob, next_state, reward, terminated in P[s][chosen_a]:
                R_pi[s] += prob * reward
                if not terminated:
                    A_mat[s, next_state] -= gamma * prob
                    
        V = np.linalg.solve(A_mat, R_pi)
        total_backups += (num_states ** 3) 
        
        policy_stable = True
        for s in range(num_states):
            old_action = policy[s]
            q_values = np.zeros(num_actions)
            
            for a in range(num_actions):
                for prob, next_state, reward, terminated in P[s][a]:
                    V_target = 0 if terminated else V[next_state]
                    q_values[a] += prob * (reward + gamma * V_target)
                    
            new_action = np.argmax(q_values)
            if old_action != new_action:
                policy_stable = False
                policy[s] = new_action
                
        if policy_stable:
            break
            
    return V, policy, iteration_count, total_backups

if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=True)
    V_pi, p_pi, it_pi, backups = policy_iteration(env.unwrapped.P, gamma=0.99)
    
    print(f"=== PART (b) REPORT ===")
    print(f"Policy Iteration completed in {it_pi} outer loop updates.")
    print(f"Total computed operational complexity metric: {backups} operations.\n")
    
    print("--- Policy Iteration V* Table (4x4 Grid) ---")
    v_grid = V_pi.reshape(4, 4)
    for row in v_grid:
        print(" | ".join(f"{val:6.4f}" for val in row))
    print("\n")
    
    print("--- Policy Iteration pi* Rendered as Arrows ---")
    arrow_map = {0: "◄", 1: "▼", 2: "►", 3: "▲"}
    holes = {5, 7, 11, 12}
    goal = 15
    
    policy_grid = p_pi.reshape(4, 4)
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
    print("=" * 24)