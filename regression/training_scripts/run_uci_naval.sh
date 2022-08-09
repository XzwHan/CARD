export EXP_DIR=./results
export N_STEPS=1000
export SERVER_NAME=a4000
export RUN_NAME=run_2
export LOSS=card_conditional
export TASK=uci_naval
export N_SPLITS=20
export N_THREADS=4
export DEVICE_ID=3

export CAT_F_PHI=_cat_f_phi
export MODEL_VERSION_DIR=card_conditional_uci_results/${N_STEPS}steps/nn/${RUN_NAME}_${SERVER_NAME}/f_phi_prior${CAT_F_PHI}
python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --run_all --n_splits ${N_SPLITS} --doc ${TASK} --config configs/${TASK}.yml #--train_guidance_only
python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --run_all --n_splits ${N_SPLITS} --doc ${TASK} --config $EXP_DIR/${MODEL_VERSION_DIR}/logs/ --test

#python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --split 7 --doc ${TASK} --config configs/${TASK}.yml #--train_guidance_only
#python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --split 8 --doc ${TASK} --config configs/${TASK}.yml #--train_guidance_only
#python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --split 9 --doc ${TASK} --config configs/${TASK}.yml #--train_guidance_only
#python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --split 10 --doc ${TASK} --config configs/${TASK}.yml #--train_guidance_only
#python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --split 11 --doc ${TASK} --config configs/${TASK}.yml #--train_guidance_only
#python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --split 12 --doc ${TASK} --config configs/${TASK}.yml #--train_guidance_only
#python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --split 13 --doc ${TASK} --config configs/${TASK}.yml #--train_guidance_only
#python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --split 14 --doc ${TASK} --config configs/${TASK}.yml #--train_guidance_only
#python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --split 15 --doc ${TASK} --config configs/${TASK}.yml #--train_guidance_only
#python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --split 16 --doc ${TASK} --config configs/${TASK}.yml #--train_guidance_only
#python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --split 17 --doc ${TASK} --config configs/${TASK}.yml #--train_guidance_only
#python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --split 18 --doc ${TASK} --config configs/${TASK}.yml #--train_guidance_only
#python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --exp $EXP_DIR/${MODEL_VERSION_DIR} --split 19 --doc ${TASK} --config configs/${TASK}.yml #--train_guidance_only
