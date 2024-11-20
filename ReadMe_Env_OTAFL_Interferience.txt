## Overview
The `SnekEnv` is a custom OpenAI Gym environment for modeling a multi-agent wireless communication system with uncertainties in channel state information (CSI). This environment simulates the interactions between base stations, relay stations, and user devices, where agents optimize their power control and beamforming strategies to minimize interference and maximize communication quality. The system is modeled using Rayleigh fading channels, path loss models, and uncertainties in channel coefficients.

This environment is designed for reinforcement learning tasks, where agents interact with the environment by taking actions that influence the system's performance. The goal is to optimize parameters like power allocation and beamforming angles to maximize the system's throughput or minimize error metrics such as Mean Square Error (MSE).

## Key Features
- **Multi-Agent Setup**: The environment simulates interactions between several base stations and relay stations, where each station (or agent) has its own power control and beamforming strategy.
- **Channel Model**: The environment uses Rayleigh fading coefficients to model channel gains between stations and users. It also incorporates path loss and uncertainties in channel state information (CSI).
- **Actions**: Agents can take actions such as adjusting transmission power, beamforming angles, and the use of certain relay stations.
- **Rewards**: Rewards are based on optimizing system metrics like MSE (Mean Square Error) and minimizing interference.
- **Customizable Parameters**: You can modify the number of stations, relay nodes, and the system configuration (e.g., distance, power constraints).

## Environment Setup

### Dependencies
- Python 3.x
- `gym`: For the environment framework.
- `numpy`: For numerical computations.
- `math`: For mathematical operations.
- `random`: For randomness in the environment.
- `collections`: For managing queues.

You can install the required dependencies with the following:

```bash
pip install gym numpy
```

### Environment Class: `SnekEnv`
The `SnekEnv` class inherits from the OpenAI Gym `Env` class, implementing the standard methods required for Gym environments (`__init__`, `reset`, `step`, `render`).

#### Key Methods:

1. **`__init__(self)`**: 
   - Initializes the environment with default parameters like the number of stations (`nk`), the number of relay stations (`nris`), and the channel gain parameters.
   - Defines the action and observation spaces.

2. **`uncertainty(self, ris, nk, hk)`**: 
   - Simulates uncertainty in the channel state information (CSI) by adding Gaussian noise to the channel coefficients.

3. **`MSE(self, eta, hk_ss, hk_ss_hat, indicator, pk, hk_ps, ppu, noise_power)`**: 
   - Calculates the Mean Squared Error (MSE) for the system.

4. **`Interference(self, hk_ss, hk_sp_hat, indicator, pk, noise_power)`**:
   - Computes the interference in the system caused by transmitting power and uncertainty in the CSI.

5. **`step(self, action)`**:
   - Takes an action (such as adjusting power control and beamforming angles), updates the system state, and returns the next state, reward, and termination condition.

6. **`reset(self)`**:
   - Resets the environment to an initial state.

#### Action Space
The action space is defined as a `Box` space where each action represents:
- Power control values for each station.
- Beamforming angles for the relay stations.
- A scalar for the scaling factor `eta`.

The action space dimensions depend on the number of stations (`nk`) and relay stations (`nris`).

#### Observation Space
The observation space represents the state of the system, including:
- Channel coefficients.
- Power control values.
- Distance metrics between stations.
- Additional parameters like noise power.

The observation space is also a `Box` space.

### Channel Modeling Functions

- **`generate_rayleigh_coefficient(num_coefficients, scale=1)`**: Generates Rayleigh fading coefficients using complex Gaussian random variables.
- **`channel_gain(nk=2, nris=3)`**: Simulates channel gains between stations and users, including the base-to-user (`hd_ss`), relay-to-user (`hd_sp`), and other relevant channel parameters.
- **`generate_path_loss(nk, dk, exp_coef=1, l=1)`**: Calculates path loss based on the distance between stations and the path loss exponent.
- **`generate_channel_coef(nk, hd, hr, G, diag, path_loss_d, path_loss_r)`**: Computes channel coefficients using path loss and channel gain values.
- **`calculate_distance(ss, original)`**: Computes the Euclidean distance between two points.
- **`distance()`**: Defines distances between various stations and users.

## Example Usage

```python
import gym
from gym import spaces
import numpy as np
import math

# Initialize the environment
env = SnekEnv()

# Reset the environment to get the initial state
state = env.reset()

# Take a random action
action = np.random.uniform(0.1, 1, env.action_space.shape)
next_state, reward, done, info = env.step(action)

print(f"Next State: {next_state}")
print(f"Reward: {reward}")
print(f"Done: {done}")
```

## Customizing the Environment
You can modify the parameters in the `__init__` method to customize the number of stations (`nk`), relay stations (`nris`), and other system parameters like noise power, power scaling factors, and path loss coefficients. Adjusting these parameters allows you to experiment with different network configurations and reinforcement learning strategies.

## Future Improvements
- Implement a more complex reward structure based on system throughput or QoS metrics.
- Incorporate different wireless models (e.g., shadow fading, non-line-of-sight conditions).
- Optimize the environment's performance by parallelizing certain computations (e.g., using GPU for matrix operations).

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
