# Notes from the papers:

*virtual loss:*

during parallel UCT (upper confidence bound tree search),
each thread that visits a state applies a temporary loss (virtual loss)
to the value of that node, to encourage threads to explore different paths.
In alphazero, it seems like they use ~8 in parallel.
This is probably tuned to run on the CPU, and 8x is the most free floating point
vectorization you get, so it makes sense to line things up to run the neural
net 8x.

*PUCT:*

Predictor UCT (upper confidence bound tree search) = PUCT.
Deterministic bandit algorithm (in the paper its PUCB).
Modification of upper confidence bound bandit algorithm (UCB1).
Additive penalty on arms with low weight, dependent on number of total pulls.
Haven't figured out how to fit the PUCB algorithm with alphazero's PUCT formula.

## Questions

virtual loss:
    What was the virtual loss used?  How was it chosen?
    Was the virtual loss non-constant?
PUCT:
    What were the c_PUCT coefficients chosen?  How were they chosen?

## TODO

AlphaZero:
- Experiment with different C_PUCT values
- Control temperature of random search
- Add noise to move selection
- Parallelize simulations (like in the paper, with virtual loss)
- Tree keeps children as a max-heap of selection value (Q + U)
- Evaluations during training
- Resignation threshold
- Find some way to amortize U(s, a) computation
- Valid move masking on Tree.select() could be improved

Games:
- Validate actions, raise on invalid (either in game, or out-of-game)
- Validate states, raise on invalid
- Maybe just have interface methods to validate state/action
- Hidden state information (not given to model)
- Add optional human-readable str(state) function

Model:
- Linear model
