Ways to run this program:

CUDA_VISIBLE_DEVICES=1 python3 train.py --input_target "Janez Jansa" --model_select EMBEDDIA/sloberta
CUDA_VISIBLE_DEVICES=1 python3 train.py --input_target "Janez Jansa" --model_select EMBEDDIA/crosloengual-bert

CUDA_VISIBLE_DEVICES=1 python3 train.py --input_target Atheism --model_select EMBEDDIA/sloberta
CUDA_VISIBLE_DEVICES=1 python3 train.py --input_target Atheism --model_select EMBEDDIA/crosloengual-bert

CUDA_VISIBLE_DEVICES=1 python3 train.py --input_target Feminism --model_select EMBEDDIA/sloberta
CUDA_VISIBLE_DEVICES=1 python3 train.py --input_target Feminism --model_select EMBEDDIA/crosloengual-bert

CUDA_VISIBLE_DEVICES=1 python3 train.py --input_target Climate --model_select EMBEDDIA/sloberta
CUDA_VISIBLE_DEVICES=1 python3 train.py --input_target Climate --model_select EMBEDDIA/crosloengual-bert

CUDA_VISIBLE_DEVICES=1 python3 train.py --input_target Abortion --model_select EMBEDDIA/sloberta
CUDA_VISIBLE_DEVICES=1 python3 train.py --input_target Abortion --model_select EMBEDDIA/crosloengual-bert