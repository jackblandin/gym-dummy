# gym-dummy

A collection of simple (dummy) OpenAI Gym environment for basic testing of RL agents.

## Fully observable environments

### `GreaterThanZero`

Only a single observation is required to predict the optimal action. Goal is to identify if the last observation is greater than zero, i.e. take action 0 if observation is < 0, and take action 1 if observation > 0.

## Partially observable environments

### `TwoInARow`

Two observations are required to predict the optimal action. Goal is to take action 1 when last two observations are the same, and take action 0 otherwise.
