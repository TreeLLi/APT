#!/bin/bash

# custom config
DATA=TODO_replace_with_data_root_path
TRAINER=APT
SHOTS=16
NCTX=16
CSC=False
CTP=end

DATASET=$1
CFG=$2

# --seed ${SEED} \


python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/${DATASET}/seed${SEED} \
--model-dir output/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/eps1_alpha0.67_step3/seed${SEED} \
--load-epoch 50 \
--eval-only \
TRAINER.COOP.N_CTX ${NCTX} \
TRAINER.COOP.CSC ${CSC} \
TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP}
