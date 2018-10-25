#! /bin/bash

models=normallogpower_models
cat $models/eval_0.01.csv | grep -Ff idx_eval_correct | cut -d',' -f2 > temp.csv
matlab -r "scoring1" -nojvm -nodisplay

