#!/bin/sh

# SEGCN
python main.py --model_name segcn --dataset twitter --learning_rate 1e-3 --l2 1e-5 --dropout 0.5 --batch_size 32
python main.py --model_name segcn --dataset lap14   --learning_rate 1e-3 --l2 1e-5 --dropout 0.5 --batch_size 16
python main.py --model_name segcn --dataset rest14  --learning_rate 1e-3 --l2 1e-4 --dropout 0.7 --batch_size 32
python main.py --model_name segcn --dataset rest15  --learning_rate 1e-3 --l2 1e-3 --dropout 0.5 --batch_size 16
python main.py --model_name segcn --dataset rest16  --learning_rate 1e-3 --l2 1e-3 --dropout 0.5 --batch_size 16

# SEGCN(14->15)
python main.py --model_name segcn_transfer --dataset rest15 --transfer 14-15 --learning_rate 1e-3 --l2 1e-3 --dropout 0.5 --batch_size 16
# SEGCN(14->16)
python main.py --model_name segcn_transfer --dataset rest16 --transfer 14-16 --learning_rate 1e-3 --l2 1e-3 --dropout 0.5 --batch_size 16
# SEGCN(15->16)
python main.py --model_name segcn_transfer --dataset rest16 --transfer 15-16 --learning_rate 1e-3 --l2 1e-3 --dropout 0.5 --batch_size 16

#SEGCN-BERT
python main.py --model_name segcn_bert --dataset twitter --learning_rate 2e-5 --l2 1e-4 --dropout 0.3 --batch_size 16
python main.py --model_name segcn_bert --dataset lap14   --learning_rate 2e-5 --l2 1e-4 --dropout 0.3 --batch_size 16
python main.py --model_name segcn_bert --dataset rest14  --learning_rate 2e-5 --l2 1e-4 --dropout 0.3 --batch_size 16
python main.py --model_name segcn_bert --dataset rest15  --learning_rate 2e-5 --l2 1e-4 --dropout 0.3 --batch_size 16
python main.py --model_name segcn_bert --dataset rest16  --learning_rate 2e-5 --l2 1e-4 --dropout 0.3 --batch_size 16
