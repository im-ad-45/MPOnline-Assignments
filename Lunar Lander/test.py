import time
import gymnasium as gym
from stable_baselines3 import PPO

def main():
    env = gym.make("LunarLander-v3", render_mode="human")
    
    model = PPO.load("models/ppo_lunarlander")
    
    for episode in range(5):
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            done = terminated or truncated

            time.sleep(0.01)
            
        print(f"Episode {episode + 1} - Score: {total_reward:.2f}")
        
    env.close()

if __name__ == "__main__":
    main()