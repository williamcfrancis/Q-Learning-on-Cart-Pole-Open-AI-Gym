# Q-Learning on the Cart-Pole problem
The Cart-Pole, also known as an inverted pendulum, is a system with a center of gravity above its pivot point. It's unstable by nature, but can be controlled by moving the pivot point under the center of mass. The goal is to keep the Cart-Pole balanced by applying appropriate forces to the pivot point.

You can try it out yourself [here](https://jeffjar.me/cartpole.html) before diving into the reinforcement learning (RL) algorithm used in this project, Q-Learning.

This project uses OpenAI Gym's version of the Cart-Pole environment, which you can find [here](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py) if you want to learn more about how the environment works.

## Run
To run the project, simply execute the `Q-Learning_Cartpole.py` file

## Results
The reward during training is plotted against the number of iterations:
![image](https://user-images.githubusercontent.com/38180831/205727954-63acdcec-9527-4439-b5df-95b126c9d2d0.png)

The reward during validation is plotted against the number of iterations:
![image](https://user-images.githubusercontent.com/38180831/205728062-099e2ce8-8713-4685-bf8c-de58a1720f57.png)

