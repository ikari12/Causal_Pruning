#!/bin/bash

# =============================================================================
# Causal Pruning Docker Execution Script
# =============================================================================
# This script launches the Docker container to run the pruning experiments.

# --- Build Command (if needed) ---
# Uncomment the line below if you need to rebuild the Docker image.
docker build -t causal-pruning .

rm -f ./results/*

# --- Main Execution Command ---
# This is the primary command to run the experiment.
docker run \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES="2,3" \
  -v "$(pwd)/results:/app/results" \
  causal-pruning

# --- Notes on other execution modes (for reference) ---
# The command above will run the default 'demo' mode.
# To run other modes, you would typically override the command like this:
#
# docker run ... causal-pruning python3 casual_pruning_execution.py --mode standard --models 4 --datasets 15
# docker run ... causal-pruning python3 casual_pruning_execution.py --mode full
