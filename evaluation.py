import gymnasium as gym
import snake_env 
from stable_baselines3 import PPO

# Load model
model_path = "models/snake_radius_final.zip"
model = PPO.load(model_path)

# Create environment with rendering
env = gym.make("Snake-v2", render_mode="human")

# Number of episodes to watch
NUM_EPISODES = 5

for ep in range(NUM_EPISODES):
    obs, _ = env.reset()
    env.render()  # Renderizar el estado inicial
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        env.render()  # Renderizar cada paso

    print(f"Episode {ep+1}: Total Reward = {total_reward}")

env.close()
 # Cerrar ventanas de OpenCV