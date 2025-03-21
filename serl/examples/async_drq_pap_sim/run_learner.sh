export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
export DISPLAY=:1 && \
export WANDB_API_KEY="afb9ea51b9652834c2840be8f209de8b38bebc34"
export CUDA_VISIBLE_DEVICES=0,4,5,7

python async_drq_pap_sim.py "$@" \
    --learner \
    --exp_name=serl_dev_drq_sim_test_resnet \
    --seed 0 \
    --training_starts 1000 \
    --critic_actor_ratio 4 \
    --encoder_type resnet-pretrained \
    --batch_size 128 \
    # --demo_path franka_lift_cube_image_20_trajs.pkl \
    # --debug # wandb is disabled when debug
