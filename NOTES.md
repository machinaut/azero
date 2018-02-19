# Notes from the papers:

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
- What were the c_PUCT coefficients chosen?  How were they chosen?

## TODO

Azero:
- Per-action update `U` instead of re-computing the whole vector every time
- Update player values when we observe opponent changing value (during backup)
- Tune how much to update based on an outcome farther down the tree
- Re-use tree between moves

Models:
- Add RBF network
- Add MLP network

Games:
- Unit test that there's no mutual information between (valid) and (state|view)
- Add 3-player tic tac toe game

Utils:
- Add tests

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
