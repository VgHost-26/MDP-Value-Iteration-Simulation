# MDP - Value Iteration Simulation

This project simulates a Markov Decision Process (MDP) on a grid, where an agent must navigate from `HOME` to `SCHOOL` while avoiding dangers.

### Main Task

- The agent must find a way from `HOME` to `SCHOOL` on a frozen lake. 
- On the lake there are multiples holes in the ice, falling into the hole will affect agent death.
- Because of slippery ice there is a change of going other direction than agent primarly decided; 0.1 for each side exluding backwards e.g. if the agent decided to go `right` there is 0.8 change going `right` and 0.1 change of going `up` and 0.1 `down`
## Features

- **Value Iteration**: Computes optimal utility values for each state.
- **Greedy Path**: Determines the best actions based on maximum utility.
- **Probabilistic Simulation**: Simulates the agent's path accounting for probabilistic slips.
- **Visualization**: Displays the greedy path and the simulation results using heatmaps and arrows.

## Environment

- **States**: 16 states (4x4 grid), where state 0 is `HOME` and state 15 is `SCHOOL`.
- **Actions**: The agent can move `left (0)`, `down (1)`, `right (2)`, `up (3)`.
- **Transition Probabilities**: Actions may result in unintended state transitions based on probabilities.

## Usage

1. Install dependencies:
   ```
   pip install numpy matplotlib seaborn
    
2.Run the script:
  ```
  python maincode.py
  ```

The output includes:
- **Greedy Path**: The best path based on value iteration.
- **Simulation _Real_ Path**: The actual path taken by the agent based on probabilities.

## Dependencies
- `numpy`
- `matplotlib`
- `seaborn`

