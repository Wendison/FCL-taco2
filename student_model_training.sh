
train_config=conf/train_pytorch_tacotron2.sa.student.yaml
tr_json=data/train_data.json
dt_json=data/val_data.json
expname=student
expdir=exp/${expname}
ngpu=1
N=0
verbose=1
seed=137
resume=""
batch_size=32

python tts_train.py \
           --ngpu ${ngpu} \
           --minibatches ${N} \
           --outdir ${expdir}/results \
           --tensorboard-dir tensorboard/${expname} \
           --verbose ${verbose} \
           --seed ${seed} \
           --resume ${resume} \
           --train-json ${tr_json} \
           --valid-json ${dt_json} \
           --config ${train_config} \
           --batch-size ${batch_size} \
           --pad-eos False \
           --use-fe-condition True \
           --append-position True \
           --use-amp True \
           --perform-KD True \
           --share-proj True
