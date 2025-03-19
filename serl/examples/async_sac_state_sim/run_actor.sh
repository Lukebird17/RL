export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05 && \
export JAX_DOWNGRADE_STABLEHLO=1 && \
export DISPLAY=:2 &&\
python async_sac_state_sim.py "$@" \
    --actor \
    --render \
    --env PandaPAP \
    --exp_name=serl_dev_sim_test \
    --seed 0 \
    --random_steps 1000 \
    --debug
