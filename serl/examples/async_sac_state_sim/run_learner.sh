export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05 && \
export JAX_DOWNGRADE_STABLEHLO=1 && \
python async_sac_state_sim.py "$@" \
    --learner \
    --env PandaPAP \
    --exp_name=serl_dev_sim_test \
    --seed 0 \
    --training_starts 1000 \
    --critic_actor_ratio 8 \
    --batch_size 256 \
    # --log_rlds_path /home/robot/SERL/serl/checkpoints
    # --debug # wandb is disabled when debug
    #--env PandaPickCube-v0 \