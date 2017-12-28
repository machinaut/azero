#!/usr/bin/env make

.PHONY: all test

all: test

test: test_alphanaughts.py
	python $^
