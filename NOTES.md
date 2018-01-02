Notes from the papers:

virtual loss:
during parallel UCT (upper confidence bound tree search),
each thread that visits a state applies a temporary loss (virtual loss)
to the value of that node, to encourage threads to explore different paths.
In alphazero, it seems like they use ~8 in parallel.
This is probably tuned to run on the CPU, and 8x is the most free floating point
vectorization you get, so it makes sense to line things up to run the neural
net 8x.

PUCT:
Predictor UCT (upper confidence bound tree search) = PUCT
Deterministic bandit algorithm (in the paper its PUCB)
Modification of upper confidence bound bandit algorithm (UCB1)
Additive penalty on arms with low weight, dependent on number of total pulls
Haven't figured out how to fit the PUCB algorithm with alphazero's PUCT formula

Questions from the papers:
    virtual loss:
        What was the virtual loss used?  How was it chosen?
        Was the virtual loss non-constant?
