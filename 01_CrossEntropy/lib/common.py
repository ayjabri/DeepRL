import torch
import torch.nn.functional as F
import time


@torch.no_grad()
def play_episode(env, net):
    state = env.reset()
    rewards = 0
    while True:
        env.render()
        state_v = torch.tensor([state], dtype=torch.float32)
        action = F.softmax(net(state_v), dim=-1).argmax().item()
        state, reward, done,_=env.step(action)
        rewards += reward
        if done:
            print(rewards)
            time.sleep(1)
            break
    env.close()
    pass