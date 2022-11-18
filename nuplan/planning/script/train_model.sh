#! /usr/bin/env bash

# Your training save dir, including checkpoints, logs, etc.
SAVE_DIR="/mnt/nas20/jingyu.qian/nuplan_training_save_dir"

# Open-loop experiment name
CL_EXPERIMENT="training_closed_loop_$(date "+%m%d")"

# Closed-loop experiment name
OL_EXPERIMENT="training_open_loop_$(date "+%m%d")"

# Cache directory
CACHE_DIR="/mnt/nas20/jingyu.qian/data/nuplan_mini_v1.1_raster_with_ego"

# Open-loop version
python nuplan/planning/script/run_training.py \
    group=$SAVE_DIR \
    experiment_name=$OL_EXPERIMENT \
    py_func=train \
    +training=training_raster_model \
    cache.cache_path=$CACHE_DIR \
    scenario_builder=nuplan_mini \
    scenario_filter=training_scenarios \
    lightning.trainer.params.accelerator=ddp_spawn \
    lightning.trainer.params.max_epochs=25 \
    lightning.trainer.params.max_time=10:00:00:00 \
    lightning.trainer.params.gpus=[0,1,2] \
    lightning.trainer.params.val_check_interval=0.5 \
    lightning.trainer.params.limit_val_batches=0.5 \
    lightning.trainer.params.sync_batchnorm=true \
    lightning.trainer.params.deterministic=true \
    lightning.trainer.params.num_sanity_val_steps=5 \
    lightning.trainer.checkpoint.resume_training="/mnt/nas20/jingyu.qian/nuplan_training_save_dir/training_open_loop_1106/2022.11.06.23.29.42/checkpoints/epoch\=24.ckpt" \
    data_loader.params.batch_size=64 \
    worker=single_machine_thread_pool \
    ~ego_controller \
    ~callbacks.closed_loop_callback \

# Closed-loop version
# python nuplan/planning/script/run_training.py \
#     group=$SAVE_DIR \
#     experiment_name=$CL_EXPERIMENT \
#     py_func=train \
#     +training=training_raster_model \
#     model.norm_layer=batchnorm \
#     cache.cache_path=$CACHE_DIR \
#     scenario_builder=nuplan_mini_cl \
#     scenario_filter=closed_loop_training_scenarios \
#     lightning.trainer.params.accelerator=ddp_spawn \
#     lightning.trainer.params.replace_sampler_ddp=False \
#     lightning.trainer.params.max_epochs=25 \
#     lightning.trainer.params.max_time=10:00:00:00 \
#     lightning.trainer.params.gpus=[0,1,2] \
#     lightning.trainer.params.val_check_interval=0.5 \
#     lightning.trainer.params.limit_val_batches=0.5 \
#     lightning.trainer.params.sync_batchnorm=true \
#     lightning.trainer.params.deterministic=true \
#     data_loader.params.batch_size=64 \
#     worker=single_machine_thread_pool \
#     ~callbacks.visualization_callback \
#     lightning.trainer.checkpoint.resume_training="/mnt/nas20/jingyu.qian/nuplan_training_save_dir/training_open_loop_1106/2022.11.06.23.29.42/checkpoints/epoch\=24.ckpt"
