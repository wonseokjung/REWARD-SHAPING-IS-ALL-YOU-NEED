import retro
from stable_baselines import PPO2
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv

env = retro.make(game='SonicAndKnuckles3-Genesis')
env = DummyVecEnv([lambda: env])

model = PPO2(policy=CnnPolicy, env=env, verbose=1,tensorboard_log ="./Users/wonseokjung/Dropbox/20_Sonic_ppo_paper/stable-baselines/1_test")
model.learn(total_timesteps=1000000)

obs = env.reset()
while True:
    action, _info = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(rewards)
    env.render()
