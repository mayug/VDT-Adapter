#!/bin/bash

# custom config
DATA=/FEAT/data/
TRAINER=ZeroshotCLIP_gpt

DATASET=$1
CFG=$2  # config file
SUBSAMPLE_CLASSES=$3
EXP=$4

for SEED in 1
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}/exp${EXP}/
    # if [ -d "$DIR" ]; then
    #     echo "Oops! The results exist at ${DIR} (so skip this job)"
    # else
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --eval-only \
    DATASET.SUBSAMPLE_CLASSES ${SUBSAMPLE_CLASSES}
    # fi
done