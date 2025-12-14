#!/bin/bash

# =============================================================================
# Causal Pruning Docker Execution Script
# =============================================================================
# This script launches the Docker container to run the pruning experiments.

# --- Build Command (if needed) ---
# Uncomment the line below if you need to rebuild the Docker image.
docker build -f Dockerfile_CWP -t causal-pruning .

#rm -f ./results/*

# --- Main Execution Command ---
# This is the primary command to run the experiment.

#docker run \
#  --gpus all \
#  -e CUDA_VISIBLE_DEVICES="0" \
#  -v "$(pwd)/results:/app/results" \
#  causal-pruning \
#  python3 casual_neuron_scoring.py

#docker run \
#  --gpus all \
#  -e CUDA_VISIBLE_DEVICES="0" \
#  -v "$(pwd)/results:/app/results" \
#  causal-pruning \
#  python3 casual_neuron_pruning.py

#docker run \
#  --gpus all \
#  -e CUDA_VISIBLE_DEVICES="0" \
#  -v "$(pwd)/results:/app/results" \
#  causal-pruning \
#  python3 casual_neuron_analysis.py

#docker run \
#  --gpus all \
#  -e CUDA_VISIBLE_DEVICES="0" \
#  -v "$(pwd)/results:/app/results" \
#  causal-pruning \
#  python3 casual_active_scoring.py

docker run \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES="0" \
  -v "$(pwd)/results:/app/results" \
  causal-pruning \
  python3 casual_active_pruning.py

#docker run \
#  --gpus all \
#  -e CUDA_VISIBLE_DEVICES="0" \
#  -v "$(pwd)/results:/app/results" \
#  causal-pruning \
#  python3 casual_wanda_scoring.py

#docker run \
#  --gpus all \
#  -e CUDA_VISIBLE_DEVICES="0" \
#  -v "$(pwd)/results:/app/results" \
#  causal-pruning \
#  python3 casual_wanda_pruning.py




##docker run \
##  --gpus all \
##  -e CUDA_VISIBLE_DEVICES="2,3" \
##  -v "$(pwd)/results:/app/results" \
##  causal-pruning \
##  python3 casual_sae_encode.py

##docker run \
##  --gpus all \
##  -e CUDA_VISIBLE_DEVICES="2,3" \
##  -v "$(pwd)/results:/app/results" \
##  causal-pruning \
##  python3 casual_sae_scoring.py

##docker run \
##  --gpus all \
##  -e CUDA_VISIBLE_DEVICES="2,3" \
##  -v "$(pwd)/results:/app/results" \
##  causal-pruning \
##  python3 casual_sae_pruning.py

docker image prune -a --force
docker container prune --force
