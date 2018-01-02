# TODO

AlphaZero:
- Experiment with different C_PUCT values
- Control temperature of random search
- Add noise to move selection
- Parallelize simulations (like in the paper, with virtual loss)
- Tree keeps children as a max-heap of selection value (Q + U)
- Evaluations during training
- Resignation threshold
- Find some way to amortize U(s, a) computation

Games:
- Validate actions, raise on invalid (either in game, or out-of-game)
- Validate states, raise on invalid
- Maybe just have interface methods to validate state/action
- Hidden state information (not given to model)

Model:
- Linear model
