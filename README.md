# gym-dummy

A collection of simple (dummy) OpenAI Gym environment for basic testing of RL agents.

## Fully observable environments

### `GreaterThanZero`

Only a single observation is required to predict the optimal action. Goal is to identify if the last observation is greater than zero, i.e. take action 0 if observation is < 0, and take action 1 if observation > 0.

### `NotXOR`

Represents the opposite of an XOR gate. Optimal policy is to take action 1 when only one of the inputs are on. This is the MDP representation of the pobs.TwoInARow, but with the last two observations flattened to a single observation.

## Partially observable environments

### `TwoInARow`

Two observations are required to predict the optimal action. Goal is to take action 1 when last two observations are the same, and take action 0 otherwise.
