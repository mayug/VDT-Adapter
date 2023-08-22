#!/bin/bash

# custom config
DATA=/FEAT/data/
TRAINER=CLIP_Adapter_gpt

DATASET=$1
CFG=$2  # config file
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=$6  # class-specific context (False or True)
SUBSAMPLE_CLASSES=$7
ADAPTER_TYPE=$8
RATIO=$9
EXP=${10}

echo "EXP: ${EXP}"
echo "RATIO: ${RATIO}"
# asd

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
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    TRAINER.CLIP_ADAPTER.WORD_ADAPTER_TYPE ${ADAPTER_TYPE} \
    TRAINER.CLIP_ADAPTER.RATIO ${RATIO} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUBSAMPLE_CLASSES}
    
    # fi
done