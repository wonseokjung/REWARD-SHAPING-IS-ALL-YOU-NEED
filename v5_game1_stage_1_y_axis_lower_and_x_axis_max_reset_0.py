import retro
from stable_baselines import PPO2
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from sonic_util import make_env

#env = retro.make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')

#env = DummyVecEnv([lambda: env])
env = DummyVecEnv([make_env])
model = PPO2(policy=CnnPolicy, env=env,verbose=1,tensorboard_log ="./v2_tensorboard/1_SonicTheHedgehod_Genesis/1_GreenHillZone_Act1/v5_game1_stage_1_y_axis_lower_and_x_axis_max_reset_0/")


model.learn(total_timesteps=2000000)

log_dir = "saving/1_GreenHillZone_Act1/"

model.save(log_dir + "v5_game1_stage_1_y_axis_lower_and_x_axis_max_reset_0")

#env.save_running_average(log_dir)


obs = env.reset()
while True:
    action, _info = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(rewards)
    env.render()
