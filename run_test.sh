PYTHON_CODE=test.py
DATAROOT=./datasets/      #/data/hohokam/Xinl/EyePACS
NAME=L2H_EyEQ
MODEL=sb
DATASET_MODE=unpaired
CHECK_PATH=./checkpoints/
STORE_PATH=./generation/
PHASE=test
MAX_TEST=15000 # test the whole test set 
METRIC_DIC=./generation/
METRIC_DIC_NAME=L2H_EyEQ_test.txt
TARGET_TRUTH_PATH=./datasets/testB
#### network structure info ###
NETG=DSCNet 
DSC_K=9
DSC_N=32
DSC_B=9
DSC_P=reflect
NCE_LAYER=0,4


python ${PYTHON_CODE} --nce_layers ${NCE_LAYER} --DSE_padding_type ${DSC_P} --DSC_n_blocks ${DSC_B} --DSC_number ${DSC_N} \
    --DSC_kernel ${DSC_K}  --netG ${NETG} --target_truth_path ${TARGET_TRUTH_PATH} \
    --metrics_dic_name ${METRIC_DIC_NAME} --metrics_save_dir ${METRIC_DIC} \
    --eval --num_test ${MAX_TEST} --phase ${PHASE} --results_dir ${STORE_PATH} --dataroot ${DATAROOT} \
    --name ${NAME} --mode ${MODEL} --dataset_mode ${DATASET_MODE} --checkpoints_dir ${CHECK_PATH}
