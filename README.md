

# Project 3: Collaboration and Competition

### Introduction
In this project, we train two agents to play tennis with one another.  More specifically, these two agents are trained to hit a ball with a racket over a net towards each other.  We use the Unity ML-Agents Tennis Environment to interact our agents with.  This environment provides observations for each agent from that agent's perspective.  Our objective is to use these observations for each episode, including their scores, to train the agents and make tennis players out of them. 

### State Space
Our state space consists of 8 continuous, floating point variables that include the position and velocity of the racket and the ball.  An agent receives a reward of +0.1 each time it hits a ball over the net.  If the ball hits the ground, the net or goes out of bounds then the agent receives a reward of -0.01.

### Action Space
Each agent controls the movement of the racket in both the horizontal and vertical direction.  The horizontal direction is moving closer to or further from the net.  Whereas, the vertical movement is jumping the racket up and down.  Both of these movement are represented by values range from -1.0 to +1.0 in continuous floating point space.

### Solving the environment
Solving this environment requires training the agents until they can reach an average maximum score of +0.5 after 100 consecutive episodes.  The average maximum score means the average over the last 100 episodes where "maximum score" means the maximum score between the two agents in the environment after one episode.

### Getting Started (using Udacity Workspace)
The Jupyter Notebook tennis.ipynb is already in the Workspace folder and loads when you enter the Udacity Workspace.  The first executable line loads all the required packages according to requirements.txt located in the included python folder.
Now running !pip -q install ./python results in two warnings:  

"ipython 6.5.0 has requirement prompt-toolkit<2.0.0,>=1.0.15, but you'll have prompt-toolkit 3.0.16 which is incompatible." 

"tensorflow 1.7.1 has requirement numpy>=1.13.3, but you'll have numpy 1.12.1 which is incompatible."

I found these warnings to be inconsequential for the project.

The tennis.ipynb notebook only takes the project as far as running six episodes with randomly generated actions.  To solve the environmnent, I created notebook tennis_t1.ipynb.  This notebook loads the Unity Tennis_Linux_NoVis/Tennis environment and contains code cells to solve the environment using the multi-agent deep deterministic policy gradient (MADDPG) approach.  After running the code cells setting up the Unity environment, I have a code cell containing function maddpg that runs the entire agent training process. 

I used two python files: a file to contain the actor and critic models, and a file containing the agent class for controlling and training the agent.  I'm using the MADDPG approach, so the file model_maddpg_1BN.py contains the class definitions of the Actor and Critic fully-connected layer models using Pytorch. The file maddpg_agent.py instantiates the multi-agent class objects, and that class instatiates the two agents used in the environment.  These two agents use the DDPG class and that class instaniates the models.  The Actor model is used to select an action based on the environment state input. The Critic model is used to generate the Q-values for the DDPG algorithm.  A lot of the code in the maddp_agent.py file comes from my last project where I train 20 agents using the DDPG algorithm.  This code can be found in [my GitHub repository](https://github.com/WestKyle/Continuous_Control).  The main difference found in maddpg_agent.py is the learn function has been modified to train two agents together instead of one at a time.  However, the actor and critic models used for ddpg is essentially the same as for maddpg except that the critic model's input is expanded to accept states and actions from both agents in this multi-agent environment.  I had tried many different variations of actor and critic models throughout this project including different configurations of batch normalization.  The "1BN" suffix of the model file indicates one batch normalization layer was used.  I had also tried two batch normalization layers and no batch normalization, but a single layer worked the best.

### Getting Started (downloading the project environment onto your PC)
If you want to run this project from you PC, then do the following:

1. Make sure you have cloned this [git repository](https://github.com/udacity/deep-reinforcement-learning)

2. Follow the instructions [here](https://github.com/udacity/deep-reinforcement-learning#dependencies) to create a new conda environment, install the python packages and create the IPython kernel.  Note that the Python Environment is installed in accordance with the package list in deep-reinforcement-learning\python\requirements.txt

3. If you have not installed Unity yet, no problem, the environment is already built for you here:

Linux: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
Mac OSX: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
Windows (32-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
Windows (64-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)


### Instructions
Follow the instructions in the jupyter notebook tennis.ipynb found in the p3_collab-compet/ folder of the DRLND GitHub repository you should have already cloned. Use the provided code cells to observe the environment state and action spaces.  Create your own code to control the agent(s) through the environment using the provided code as an example to guide you and your chosen DRL methods. 

For my project, you can find my Jupyter Notebook in the repository named tennis_t1.ipynb.  All the code cells are provided in this notebook for running the project and solving the environment. You can use this Jupiter notebook in the Udacity workspace.  To use it on your own PC, you will need to change the Unity environment to the one that corresponds with your operating system as listed above in "Getting Started".
