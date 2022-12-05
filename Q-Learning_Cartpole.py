import gym
import numpy as np
import torch as th
import torch.nn as nn
import random
import matplotlib.pyplot as plt

def rollout(e, q, eps=0, T=200):
    traj = []
    # Reset environment and get initial state
    x = e.reset()
    for t in range(T):
        # Get action from policy (q network)
        u = q.control(th.from_numpy(x).float().unsqueeze(0), eps=eps)
        # u = u.int().numpy().squeeze()
        # Execute action in the environment
        xp, r, d, info = e.step(u)
        t = dict(x=x, xp=xp, r=r, u=u, d=d, info=info)
        traj.append(t)
        # Update current state
        x = xp
        # If done, terminate rollout
        if d:
            break
    return traj

class q_t(nn.Module):
    def __init__(s, xdim, udim, hdim=16):
        super().__init__()
        s.xdim, s.udim = xdim, udim
        s.m = nn.Sequential(
            nn.Linear(xdim, hdim),
            nn.ReLU(True),
            nn.Linear(hdim, hdim),
            nn.ReLU(True),
            nn.Linear(hdim, udim),
        )

    def forward(s, x):
        return s.m(x)

    def control(s, x, eps=0.1): 
        # Get q values for all controls
        q = s.m(x)

        ### TODO: XXXXXXXXXXXX
        # eps-greedy strategy to choose control input
        # note that for eps=0 you should return the correct control u
        random_a = np.random.random()
        if random_a < 1 - eps:
            u = q.argmax().item() #epsilon greedy
        else:
            u = np.random.randint(0, 2) #explore
        return u

def loss(q, ds, q_target):
    ### TODO: XXXXXXXXXXXX
    # 1. sample mini-batch from datset ds
    # 2. code up dqn with double-q trick
    # 3. return the objective f
    batch_size = 64
    loss = nn.MSELoss()

    i = 0
    x_l = []
    xp_l = [] 
    r_l = []
    u_l = []

    while True:
        if i > batch_size - 1:
            break
        idx = random.randint(0, len(ds) - 1)
        idx1 = random.randint(0, len(ds[idx]) - 1)
        if ds[idx][idx1]['d']:
            continue
        x_l.append(list(ds[idx][idx1]['x']))
        xp_l.append(list(ds[idx][idx1]['xp']))
        r_l.append(ds[idx][idx1]['r'])
        u_l.append(ds[idx][idx1]['u'])
        i += 1

    x = th.from_numpy(np.array(x_l)).float()
    xp = th.from_numpy(np.array(xp_l)).float()
    r = th.from_numpy(np.array(r_l)).float().view(batch_size, 1)
    u = th.from_numpy(np.array(u_l)).view(batch_size, 1)
    pred = q(x).gather(1, u)
    q_a=th.argmax(q(xp).detach(), dim=1).reshape(-1,1)
    q_1=q_target(xp).detach().gather(1,q_a)
    t = r + 0.9 * q_1.max(1)[0].view(batch_size, 1)

    f = loss(pred, t)
    return f

def evaluate(q):
    ### TODO: XXXXXXXXXXXX
    # 1. create a new environment e
    # 2. run the learnt q network for 100 trajectories on this new environment
    # to take control actions. Remember that you should not perform
    # epsilon-greedy exploration in the evaluation phase
    # 3. report the average discounted return of these 100 trajectories
    e=gym.make('CartPole-v1')
    x=e.reset()
    traj=[]
    rs=[]
    for t in range(100):
        u = q.control(th.from_numpy(x).float().unsqueeze(0), 0.1)
        # u = u.int().numpy().squeeze()

        xp, r, d, info = e.step(u)
        t = dict(x=x, xp=xp, r=r, u=u, d=d, info=info)

        x = xp
        traj.append(t)
        rs.append(r)
        if d==True:
            # print(len(traj))
            break

    avgdisc = sum([rr * 0.9 ** k for k, rr in enumerate(rs)])
    return avgdisc

if __name__=='__main__':
    # Create environment
    e = gym.make('CartPole-v1')
    xdim, udim = e.observation_space.shape[0], e.action_space.n

    # Create q network
    q = q_t(xdim, udim, 8)
    optim = th.optim.Adam(q.parameters(), lr=1e-3, weight_decay=1e-4)

    # Dataset of trajectories
    ds = []
    q_target = q_t(xdim, udim, 8)

    # Collect few random trajectories with eps=1
    for i in range(1000):
        ds.append(rollout(e, q, eps=1, T=200))
    lossl = []
    trainret = []
    evalret = []

    for i in range(1000):
        q.train()
        t = rollout(e, q)
        ds.append(t)

        # Perform weights updates on the q network
        # need to call zero grad on q function to clear the gradient buffer
        q.zero_grad()
        f = loss(q, ds, q_target)
        lossl.append(f.item())
        f.backward()
        optim.step()

        # Exponential averaging for the target

        if i % 10 == 0:
            q_target.load_state_dict(q.state_dict())
    
        evalret.append(evaluate(q))
        trainret.append(len(t))
    
    
    # plt.plot(lossl)
    # plt.savefig('loss.png')
    # plt.close()

    # plt.plot(evalret)
    # plt.savefig('evaluate.png')
    # plt.close()

    # plt.plot(trainret)
    # plt.savefig('train.png')
    # plt.close()
        # print('Logging data to plot')

