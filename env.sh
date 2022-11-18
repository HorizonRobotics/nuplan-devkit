#!/bin/bash
NUPLAN_REPO=$(pwd)
export PYTHONPATH=$PYTHONPATH:$NUPLAN_REPO
export NUPLAN_DATA_ROOT="/mnt/nas20/nuplanv1.1/data/cache/"
export NUPLAN_MAPS_ROOT="/mnt/nas20/nuplanv1.1/maps"
export NUPLAN_EXP_ROOT="$NUPLAN_REPO/exp"