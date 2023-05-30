python ./src/main.py \
--method ${METHOD} --mode stream \
--model_ckpt_path ./pretrain_ckpts/pretrain/ \
--pretrain_ddfm_model_ckpt_path ./pretrain_ckpts/ddfm/ \
--data_path ./data/criteo/data.txt --epoch 1 \
--lr ${LR} --weight_decay ${WC} \
> logger/${METHOD}/criteo.log