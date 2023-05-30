# pretrain base model
python ./src/main.py \
--method Pretrain --mode pretrain --model_ckpt_path ./pretrain_ckpts/pretrain/ \
--data_path ./data/criteo/data.txt \
> logger/Pretrain/criteo.log

# pretrain ddfm
python ./src/main.py \
--method DDFM --mode pretrain --pretrain_ddfm_model_ckpt_path ./pretrain_ckpts/ddfm/ \
--data_path ./data/criteo/data.txt \
> logger/DDFM/criteo_pretrain.log