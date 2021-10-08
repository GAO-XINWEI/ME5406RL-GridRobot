# RL-GridRobot
This is a simple project for agents moving in gride world. Table based and none neural network contained.

## Topic
1. Monto Carlo Control
2. SARSA
3. Q-Learning

## Requirement
1. gym==0.19.0
2. matplotlib==3.3.4
3. numpy==1.19.5
4. pyglet==1.5.19

## Run Insturction
Please follow the instruction to run the program on WIN:
1. Run 'pip install -r requirements.txt' in pyhton36 environment.
2. Replace the '__init__.py' in '/Env_Name/Lib/site-packages/gym/'.
3. Add the 'ME5406' folder in 'Env_Name/Lib/site-packages/gym/envs/'.

After that the 'main.py' is ready to run.

# Environment Setting
## The render is looks like:
- ![f2](https://user-images.githubusercontent.com/76041828/136560019-cdec08a0-aab4-4ddc-8420-ea19107f3ee3.png)

## The state table of the gride world:
- ![f1](https://user-images.githubusercontent.com/76041828/136560036-87612544-121a-4117-9933-a901a7292777.png)

# Result
## Train result and the updating process of three algorithoms on two different tasks:
-![f3](https://user-images.githubusercontent.com/76041828/136559888-d4029d94-e051-41f1-836e-c03d83cf7610.png)
## The maxQ in gride 16.
- ![task1_qlearn](https://user-images.githubusercontent.com/76041828/136559808-4eab47f1-1829-48fc-827d-addc8e2f8e6f.png)
## The maxQ in gride 100.
- ![task2_sarsa](https://user-images.githubusercontent.com/76041828/136559828-9f713efe-5213-4569-96ee-9c61540e4254.png)

