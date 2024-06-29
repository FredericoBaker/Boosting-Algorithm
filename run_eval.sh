#!/bin/bash

PYTHON_PATH="python"
SCRIPT="main.py"
FILEPATH="data/tic-tac-toe.data"
TEST_SIZE=0.2
K_FOLDS=5
MAX_ITERATIONS=400

$PYTHON_PATH $SCRIPT --filepath $FILEPATH --test_size $TEST_SIZE --k_folds $K_FOLDS --max_iterations $MAX_ITERATIONS
