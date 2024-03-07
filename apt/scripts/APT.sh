#!/bin/bash

# custom config
DATA=TODO_replace_with_data_root_path
TRAINER=APT

DATASET=$1
CFG=$2  # config file
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=$6  # class-specific context (False or True)
EPS=$7  # epsilon for AT
ALPHA=$8  # alpha or step size for AT
STEPS=$9  # number of steps for AT
SEED=${10}
ATP=${11}
PALPHA=${12}


if [ ${ATP} == 'perturbed' ]
then
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/eps${EPS}_alpha${ALPHA}_step${STEPS}_${ATP}_${PALPHA}/seed${SEED}
elif [ ${ATP} == 'constant' ]
then
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/eps${EPS}_alpha${ALPHA}_step${STEPS}_${ATP}/seed${SEED}
else
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/eps${EPS}_alpha${ALPHA}_step${STEPS}/seed${SEED}
fi


if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python train.py \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --eps ${EPS} \
    --alpha ${ALPHA} \
    --steps ${STEPS} \
    --adv-prompt ${ATP} \
    --prompt-alpha ${PALPHA} \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.NUM_SHOTS ${SHOTS}
fi
