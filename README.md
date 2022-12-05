# Q-Learning on the Cart-Pole problem
Cartpole - known also as an Inverted Pendulum is a pendulum with a center of gravity above its pivot point. It’s unstable, but can be controlled by moving the pivot point under the center of mass. The goal is to keep the cartpole balanced by applying appropriate forces to a pivot point.

[Try it out yourself here](https://jeffjar.me/cartpole.html) before we burden our RL algorithm, Q-Learning.

I used OpenAI Gym’s version of CartPole, read the code at [OpenAI Gym Cartpole](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py) to understand the environment.

## Results
The reward during training is plotted against the number of iterations:
![image](https://user-images.githubusercontent.com/38180831/205727954-63acdcec-9527-4439-b5df-95b126c9d2d0.png)

The reward during validation is plotted against the number of iterations:
![image](https://user-images.githubusercontent.com/38180831/205728062-099e2ce8-8713-4685-bf8c-de58a1720f57.png)

