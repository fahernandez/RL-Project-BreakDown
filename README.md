# Atari Breakout with DRL Gradient Policies.
This project aims to measure the impact of different reward attribution strategies on an agent's learning process while playing Atari Game Breakout. The agent will be trained using the Deep Reinforcement Learning Algorithm Reinforce Policy Gradient.
The impact of the reward attribution strategy will be measured by the number of points received by the agent at the end of 1000 consecutive plays. Four experiments will be made varying the reward attribution strategery, and each experiment will be repeated five times (20 000 consecutive plays will be executed-5 per experiment). 
The project will conclude with the strategy with the best average performance; thus, an adequate approach while designing these kinds of agents.

## Environment details
1. Game: Atari Breakout [ALE/Breakout-v5](https://gym.openai.com/envs/Breakout-v0/)
2. Output: grayscale images.
3. Frame Skip 4. Frame skip controls how many frames an action will be repeated. More detail [here](https://www.ijcai.org/Proceedings/2018/0787.pdf)
4. Lives per episode: 3
5. Episodes per experiment 1000

### Environment output pipeline
Before using the environment output to train the model a preprocessing of the output
is made to reduce noise. Here the detail of this task
1. Input: 260x160 grayscale images.
2. Cut all figures not related to the game interaction.
3. Downsize the image by x2
4. Set background color to zero.
5. Output: 6400 vector (80x80 image) 

## Agent Details
1. Agent action space: Not move, Hit the ball, Move Left, Move Right
2. Point system: Amount of bricks on the game broken.

### Reward system.
1. +n points by breaking the bricks.
2. -1 by losing lives (this varies per experiment)
3. Error attribution: Discounted error with a factor of 0.99 (REINFORCE algorithm) (this also varies per experiment)

#### Experiments
1. 4 experiments were executed varying the game reward attribution (reward attribution strategy)
strategy. Reward strategy:
   1. Punish Agent: The agent will be punished -1 each time a live is lost in the game. This is not the Atari present implementation.
   2. Normalized rewards: The accumulative reward attribution will be z-score normalized.

| Experiment # | Punish Agent | Normalized rewards |
|--------------|--------------|--------------------|
| 1            | No           | No                 |
| 2            | Yes          | No                 |
| 3            | No           | Yes                |
| 4            | Yes          | Yes                |
2. Each experiment was executed 5 times.
3. 20 000 episodes will be played in total.

### Training process.
1. Each of the environment output image are feed to a CNN whose output is the next action to take in the game.
2. CNN details:
   1. Input: 80x80 image
   2. First Layer-Convolutional Layer:
         1. Filter: 64
         2. Kernel size: 8
         3. Activation layer: RELU
         4. Strides: (4,4)
   3. Second Layer-Convolutional NN:
      1. Filter: 32
      2. Kernel size: 4
      3. Activation layer: RELU
      4. Strides: (2,2)
   4. Third Layer Convolutional NN:
      1. Filter: 32
      2. Kernel size: 3
      3. Activation layer: RELU
      4. Strides: (1,1) 
   5. Four Layer fully connected dense layer
      1. 200 units
      2. Activation layer: RELU
      3. L1/L2 Regularization
      4. Glorot Uniform initialization
   6. Output layer
      1. 4 units
      2. Activation layer: SOFTMAX
   7. Adam Optimizer with a learning rate of 0.001
   8. L1/L2 regularization over the weights of 0.001.
   9. Loss:
      1. The loss function was implemented following the REINFORCE Algorithm. More detail in the project report.
   
# Requirements
1. Install Python >= 3.8
2. Install all library requirements on requirements.txt
3. Optional (Install Cuda Support for faster execution)

# How to use this project.
1. Execute main.py to run all the experiments.
2. Navigate on analysis.ipynb to see the analysis of the information shown in the reports.
3. Execute play.py to see the best agent playing.
4. Experiment results are stored in folder execution.

# Acknowledgement
The game is simulated through the Arcade Learning Environment [ALE], which uses the Stella [Stella] Atari emulator.

1. [ALE]	MG Bellemare, Y Naddaf, J Veness, and M Bowling. "The arcade learning environment: An evaluation platform for general agents." Journal of Artificial Intelligence Research (2012). https://github.com/mgbellemare/Arcade-Learning-Environment
2. [Stella]	Stella: A Multi-Platform Atari 2600 VCS emulator https://stella-emu.github.io/

# Notes
1. The environment and agent based setup was based on the recommended settings by [x,y](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
). Some minor details were made to adapt the experiment to the host environment.

# Troubleshooting
## Install Cuda package for TF
1. sudo apt install nvidia-cuda-toolkit
2. Install Cuda https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

## Solution to display drivers
1. https://askubuntu.com/questions/1352158/libgl-error-failed-to-load-drivers-iris-and-swrast-in-ubuntu-20-04