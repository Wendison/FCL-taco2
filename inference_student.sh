nj=50  # number of splits

python splitjson.py --parts $nj data/test_data.json

exp_name=student
outdir=exp/${exp_name}
expdir=exp/${exp_name}
name=test-results-pred-100
model=snapshot.ep.100
python tts_decode.py \
          --test-teacher False \
          --ngpu 1 \
          --verbose 1 \
          --out ${outdir}/${name}/feats.1 \
          --json data/split${nj}utt/test_data.1.json \
          --model ${expdir}/results/${model} \
          --model-conf ${expdir}/results/model.json \
          --pad-eos False

parallel-wavegan-decode \
    --checkpoint vocoder/PWG/PWG.pkl \
    --feats-scp ${outdir}/${name}/feats.1.scp \
    --outdir ${outdir}/${name}-pwg-cmd
