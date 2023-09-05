python scripts/mycode/mypredict.py \
    --model bert-large/best_checkpoint_val_f1=0.7626_epoch=018.ckpt \
    --processor bert-large/processor_config.json \
    --model_input data/preprocessed/semeval2007/semeval2007.json \
    --evaluation_input data/original/semeval2007/semeval2007.gold.key.txt \
    --n_Tsamples 2 \
    --rand_seed 10 \
    --mark K07 