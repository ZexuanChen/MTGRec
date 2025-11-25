

DATASET=Musical_Instruments
TOKEN=rqvae/sentence-t5-base_256,256,256,256

L=6
CKPT=./ckpt/Musical_Instruments/Nov-19-2025_05-32-b9dbc8
CKPT_EPOCHS=(90 100 110 120)
SEM_IDS=(9986,9987,9988,9989,9990,9991)


for CKPT_EPOCH in "${CKPT_EPOCHS[@]}"; do
for SEM_ID in "${SEM_IDS[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1,2 \
    accelerate launch \
        --main_process_port 25232 \
        --num_processes 3 finetune.py \
        --dataset=${DATASET} \
        --config_file=config/ftconfig.yaml \
        --token_prefix=${TOKEN} \
        --lr=0.0002 \
        --warmup_steps=0 \
        --num_layers=$L \
        --num_decoder_layers=$L \
        --epochs=100 \
        --patience=10 \
        --pretrained_model=${CKPT}_${CKPT_EPOCH}.pth \
        --sem_id_epochs=[$SEM_ID]
done
done





DATASET=Industrial_and_Scientific
TOKEN=rqvae/sentence-t5-base_256,256,256,256

L=5
CKPT=./ckpt/Industrial_and_Scientific/Nov-19-2025_22-00-ba0854
CKPT_EPOCHS=(90 100 110 120)
SEM_IDS=(9991,9992,9993,9994,9995)


for CKPT_EPOCH in "${CKPT_EPOCHS[@]}"; do
for SEM_ID in "${SEM_IDS[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1,2 \
    accelerate launch \
        --main_process_port 25232 \
        --num_processes 3 finetune.py \
        --dataset=${DATASET} \
        --config_file=config/ftconfig.yaml \
        --token_prefix=${TOKEN} \
        --lr=0.0002 \
        --warmup_steps=0 \
        --num_layers=$L \
        --num_decoder_layers=$L \
        --epochs=100 \
        --patience=10 \
        --pretrained_model=${CKPT}_${CKPT_EPOCH}.pth \
        --sem_id_epochs=[$SEM_ID]
done
done





DATASET=Video_Games
TOKEN=rqvae/sentence-t5-base_256,256,256,256

L=7
CKPT=./ckpt/Video_Games/Nov-20-2025_18-46-8bf284
CKPT_EPOCHS=(140 150 160)
SEM_IDS=(9986,9987,9988,9989,9990,9991)


for CKPT_EPOCH in "${CKPT_EPOCHS[@]}"; do
for SEM_ID in "${SEM_IDS[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1,2 \
    accelerate launch \
        --main_process_port 25232 \
        --num_processes 3 finetune.py \
        --dataset=${DATASET} \
        --config_file=config/ftconfig.yaml \
        --token_prefix=${TOKEN} \
        --lr=0.0002 \
        --warmup_steps=0 \
        --num_layers=$L \
        --num_decoder_layers=$L \
        --epochs=100 \
        --patience=10 \
        --pretrained_model=${CKPT}_${CKPT_EPOCH}.pth \
        --sem_id_epochs=[$SEM_ID]
done
done
