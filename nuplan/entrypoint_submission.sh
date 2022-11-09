#!/bin/bash

[ -d "/mnt/data" ] && cp -r /mnt/data/nuplan-v1.0/maps/* $NUPLAN_MAPS_ROOT

# Modify `planner=simple_planner` to submit your planner instead.
# For an example of how to write a hydra config, see nuplan/planning/script/config/simulation/planner/simple_planner.yaml.
python nuplan/planning/script/run_submission_planner.py output_dir=/tmp/ planner=simple_planner
