from osim.env import OsimEnv

env = OsimEnv(visualize=False)

observation = env.reset()
for i in range(200):
    observation, reward, done, info = env.step(env.action_space.sample())
    if done:
        env.reset()
        print(i)
print(observation)