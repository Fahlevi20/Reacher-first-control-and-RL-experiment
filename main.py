import gymnasium as gym 
import numpy as np

env  = gym.make("Reacher-v5", render_mode="human")

observation, info = env.reset()

velocity = observation[6:8] 
print("Velocity after reset:", velocity)

for step in range(1000):
    action = np.array([0.05, -0.10], dtype=np.float32)
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        print("Episode finished after", step + 1, "steps")
        print("velocity at end:", observation[6:8])
        print("Fingertip X and Y position is close` to target:", observation[8:10])
        break

print(info)
print("Observation:", observation)
print("Number of values:", len(observation))
print("Observation space:", env.observation_space)

env.close()
