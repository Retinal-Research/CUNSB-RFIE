PYTHON_CODE=train.py
DATAROOT=./datasets/
NAME=L2H_EyEQ  ### new experiment for unpaired SB_training
MODEL=sb
WEIGHT_SB=1.0
GPU_ID=0
DATASET_MODE=unpaired
BATCH_SIZE=8
DIRECTION=AtoB
WEB_DISPLAY=-1 
SAVE_EPOCH_FRE=10
PRINT_FRE=688 
SAVE_LATEST_FRE=2000
DISPLAY_FRE=688 
N_WORKER=4
CHECK_PATH=./checkpoints/
## network para
NORM_DIS=batch  
### validation control
IF_VAL=True  ### 
VAL_DATA_PATH=./datasets/
VAL_BATCH=16
###sb_model control  for regularization
# SSIM loss
IF_SSIM=True
WEIGHT_SSIM=0.8
NCE_IDT=True  ### control ssim_idt and nce_idt
WEIGHT_NCE=1.0
SSIM_IDT=True  ## should be the same as nce_idt

## DSCNet parameters
G_TYPE=DSCNet
DSC_K=9
DSC_N=32
DSC_B=9
DSC_P=reflect
NCE_LAYER=0,4








python ${PYTHON_CODE}  --DSE_padding_type ${DSC_P} --DSC_n_blocks ${DSC_B} --DSC_number ${DSC_N} --DSC_kernel ${DSC_K} \
 --nce_layers ${NCE_LAYER} --netG ${G_TYPE} --nce_idt ${NCE_IDT} --lambda_ssim ${WEIGHT_SSIM} \
 --if_ssim ${IF_SSIM} --ssim_idt ${SSIM_IDT} --validation_batch ${VAL_BATCH} --validation_phase ${IF_VAL} --validation_dict_path ${VAL_DATA_PATH}  \
 --if_validation ${IF_VAL} --normD ${NORM_DIS}  --dataroot ${DATAROOT} --name ${NAME} --mode ${MODEL} --lambda_SB ${WEIGHT_SB} \
 --lambda_NCE ${WEIGHT_NCE} --gpu_ids ${GPU_ID} --dataset_mode ${DATASET_MODE} --batch_size ${BATCH_SIZE} --direction ${DIRECTION} \
    --display_id ${WEB_DISPLAY} --save_epoch_freq ${SAVE_EPOCH_FRE} \
 --print_freq ${PRINT_FRE} --save_latest_freq ${SAVE_LATEST_FRE} --display_freq ${DISPLAY_FRE} --num_threads ${N_WORKER} --checkpoints_dir ${CHECK_PATH}
