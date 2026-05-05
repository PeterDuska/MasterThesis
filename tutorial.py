from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.heuristics import ChargeAsFastAsPossible
import matplotlib.pyplot as plt

config_file = "ev2gym/example_config_files/V2GProfitPlusLoads.yaml"

env = EV2Gym(
    config_file=config_file, 
    save_replay=True, 
    save_plots=True
)

state, info = env.reset()

print("Environment loaded.")
print("Simulation length:", env.simulation_length)
print("Initial state type:", type(state))

agent = ChargeAsFastAsPossible()
total_reward = 0

for t in range(env.simulation_length):
    actions = agent.get_action(env)
    new_state, reward, done, truncated, stats = env.step(actions)
    total_reward += reward

    if done or truncated:
        break

print("Total reward:", total_reward)
print("Stats:", stats)