#!/usr/bin/env make


.PHONY: all test cprof lprof

all: test

test: test.py
	python $^

cprof: azero.py
	python -m cProfile -s cumtime azero.py > $^.cprof
	head -20 < $^.cprof

lprof: azero.py
	kernprof -l -b $^
	python -m line_profiler $^.lprof

shell: azero.py
	ipython -i $^
