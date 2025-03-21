export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
export DISPLAY=:1 && \
export WANDB_API_KEY="afb9ea51b9652834c2840be8f209de8b38bebc34"
export CUDA_VISIBLE_DEVICES=0,4,5,7
python async_drq_pap_sim.py "$@" \
    --actor \
    # --render \
    --exp_name=serl_dev_drq_sim_test_resnet \
    --seed 0 \
    --random_steps 1000 \
    --encoder_type resnet-pretrained \
    --debug
