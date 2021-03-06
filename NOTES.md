# NOTES

## Stack of things to do

- Conv2d Model
- Loading saved models
- faster rollouts
    - virtual loss & batching
    - numpy forward


## Notes from the papers:

virtual loss:
- during parallel UCT (upper confidence bound tree search),
- each thread that visits a state applies a temporary loss (virtual loss)
    to the value of that node, to encourage threads to explore different paths.
- In alphazero, it seems like they use ~8 in parallel.
- This is probably tuned to run on the CPU, and 8x is the most free floating point
    vectorization you get, so it makes sense to line things up to run the neural
    net 8x.

PUCT:
- Predictor UCT (upper confidence bound tree search) = PUCT.
- Deterministic bandit algorithm (in the paper its PUCB).
- Modification of upper confidence bound bandit algorithm (UCB1).
- Additive penalty on arms with low weight, dependent on number of total pulls.
- Haven't figured out how to fit the PUCB algorithm with alphazero's PUCT formula.

## Questions

virtual loss:
- What was the virtual loss used?  How was it chosen?
- Was the virtual loss non-constant?

PUCT:
- How was c_PUCT chosen?

Noise:
- Why use Dirichlet noise?  Why not other (simple) kinds?

## TODO

Azero:
- Update search to use n_player length value predictions from the model
- Per-action update `U` instead of re-computing the whole vector every time
- Update player values when we observe opponent changing value (during backup)
- Tune how much to update based on an outcome farther down the tree
- Re-use tree between moves
- Do something useful with first value from evaluating tree at root node
- Find some way to incorporate the other players values into the MCTS selection

Models:
- Load models from files
- Test overfitting
- Add update and tests
- Add RBF network
- Add Boltzmann machine
- Add random forests

NN:
- Add dropout
- Add batchnorm

Games:
- make more TODO items

Utils:
- Add tests

Hypers:
- c_PUCT - constant determining level of exploration in tree search
    - (AlphaGo had 5.0, AlphaGoZero and AlphaZero unlisted)
    - Benchmark tree-search by itself and see how this affects results
    - Try to keep this constant -- annealing this seems tricky
- tau - temperature controlling exploration in game
    - AlphaGoZero had 1.0 for first 30 moves then "infinitesimal" -> 0
    - See how this affects play strength given fixed search results
    - See how this affects training (regularizes?)
    - Anneal this over training? 1.0 -> 0.0
- simulation games per search
    - Varied by game in Alpha{Go}Zero
    - Possibly intelligently tune over training?
    - Would be good to measure how much more information each one gives us
- loss function on value predictions
    - AlphaGoZero uses MSE, I think black value = -white value here
    - For multi-player value predictions, I think L2 is a good place to start
    - Cosine loss might also work well, given sum(all rewards) = constant
    - Notably *not* using crossentropy, because not interpreting value as prob
- Exploration noise
    - AlphaZero added Dirichlet noise, maybe try implementing that first
- Value bias - verify adding a constant to all values doesn't change results

## Old TODO

AlphaZero:
- Unroll stack-based simulations to be iterative
- Handle value change-of-sign with subtree move
- value is value-to-player1, -value is value-to-player2, player1=1, player2=-1
    - value-to-player = value * player
- Experiment with different C_PUCT values
- Control temperature of random search
- Add noise to move selection
- Parallelize simulations (like in the paper, with virtual loss)
- Tree keeps children as a max-heap of selection value (Q + U)
- Resignation threshold
- Find some way to amortize U(s, a) computation
- Valid move masking on Tree.select() could be improved
- Trees can be sparse in valid actions (instead of storing values for invalids)
- define __slots__ for classes

Games:
- Tic Tac Two: variant where each player gets two moves in a row
- Validate actions, raise on invalid (either in game, or out-of-game)
- Validate states, raise on invalid
- Maybe just have interface methods to validate state/action
- Hidden state information (not given to model)
- Add optional human-readable str(state) function
- Handle games where moves don't always alternate (e.g. checkers double jump)

Model:
- Linear model
