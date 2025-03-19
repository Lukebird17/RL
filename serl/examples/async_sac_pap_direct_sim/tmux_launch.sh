#!/bin/bash

EXAMPLE_DIR=${EXAMPLE_DIR:-"examples/async_sac_pap_direct_sim"}
CONDA_ENV=${CONDA_ENV:-"serl"}
export CUDA_HOME=/usr/local/cuda-12:$CUDA_HOME 
export PATH=/usr/local/cuda-12:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH

cd $EXAMPLE_DIR
echo "Running from $(pwd)"

# Create a new tmux session
tmux new-session -d -s serl_direct_session

# Split the window vertically
tmux split-window -v

# Navigate to the activate the conda environment in the first pane
tmux send-keys -t serl_direct_session:0.0 "conda activate $CONDA_ENV && bash run_actor.sh" C-m

# Navigate to the activate the conda environment in the second pane
tmux send-keys -t serl_direct_session:0.1 "conda activate $CONDA_ENV && bash run_learner.sh" C-m

# Attach to the tmux session
tmux attach-session -t serl_direct_session

# kill the tmux session by running the following command
# tmux kill-session -t serl_session
