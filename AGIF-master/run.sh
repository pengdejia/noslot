###################################         train   ########################################
# MixATIS dataset
CUDA_VISIBLE_DEVICES=0 python train.py -g -bs=16 -ne=100 -dd=./data/MixATIS -lod=./log/MixATIS -sd=./save/MixATIS -nh=4 -wed=32 -sed=128 -ied=64 -sdhd=64 -dghd=64 -ln=MixATIS.txt

# MixATIS_clean dataset
#CUDA_VISIBLE_DEVICES=0 python train.py -g -bs=16 -ne=200 -dd=./data/MixATIS_clean -lod=./log/MixATIS_clean -sd=./save/MixATIS_clean -nh=4 -wed=32 -sed=128 -ied=128 -sdhd=128 -dghd=64 -ln=MixSNIPS_clean.txt 

# MixSNIPS dataset
#CUDA_VISIBLE_DEVICES=0 python train.py -g -bs=64 -ne=50 -dd=./data/MixSNIPS -lod=./log/MixSNIPS -sd=./save/MixSNIPS -nh=8 -wed=32 -ied=64 -sdhd=64 -ln=MixSNIPS.txt

# MixSNIPS_clean dataset
#CUDA_VISIBLE_DEVICES=0 python train.py -g -bs=64 -ne=100 -dd=./data/MixSNIPS_clean -lod=./log/MixSNIPS_clean -sd=./save/MixSNIPS_clean -nh=8 -wed=32 -ied=64 -sdhd=64 -ln=MixSNIPS_clean.txt

# ATIS dataset
#CUDA_VISIBLE_DEVICES=0 python train.py -g -bs=16 -ne=300 -dd=./data/ATIS -lod=./log/ATIS -sd=./save/ATIS -nh=4 -wed=64 -ied=128 -sdhd=128 -ln=ATIS.txt

# SNIPS dataset
#CUDA_VISIBLE_DEVICES=0 python train.py -g -bs=16 -ne=200 -dd=./data/SNIPS -lod=./log/SNIPS -sd=./save/SNIPS -nh=8 -wed=64 -ied=64 -sdhd=64 -ln=SNIPS.txt 

####################################          test    ########################################


# MixATIS dataset
CUDA_VISIBLE_DEVICES=0 python train.py -g -bs=16 -ne=0 -dd=./data/MixATIS -lod=./log/MixATIS -sd=./save/best/MixATIS -ld=./save/best/MixATIS -nh=4 -wed=32 -sed=128 -ied=64 -sdhd=64 -dghd=64 -ln=MixATIS.txt

# MixATIS_clean dataset
#CUDA_VISIBLE_DEVICES=0 python train.py -g -bs=16 -ne=0 -dd=./data/MixATIS_clean -lod=./log/MixATIS_clean -sd=./save/best/MixATIS_clean -ld=./save/best/MixATIS_clean -nh=4 -wed=32 -sed=128 -ied=128 -sdhd=128 -dghd=64 -ln=MixSNIPS_clean.txt 

# MixSNIPS dataset
#CUDA_VISIBLE_DEVICES=0 python train.py -g -bs=64 -ne=0 -dd=./data/MixSNIPS -lod=./log/MixSNIPS -sd=./save/best/MixSNIPS -ld=./save/best/MixSNIPS -nh=8 -wed=32 -ied=64 -sdhd=64 -ln=MixSNIPS.txt

# MixSNIPS_clean dataset
#CUDA_VISIBLE_DEVICES=0 python train.py -g -bs=64 -ne=0 -dd=./data/MixSNIPS_clean -lod=./log/MixSNIPS_clean -sd=./save/best/MixSNIPS_clean -ld=./save/best/MixSNIPS_clean -nh=8 -wed=32 -ied=64 -sdhd=64 -ln=MixSNIPS_clean.txt

# ATIS dataset
#CUDA_VISIBLE_DEVICES=0 python train.py -g -bs=16 -ne=0 -dd=./data/ATIS -lod=./log/ATIS -sd=./save/best/ATIS -ld=./save/best/ATIS -nh=4 -wed=64 -ied=128 -sdhd=128 -ln=ATIS.txt

# SNIPS dataset
#CUDA_VISIBLE_DEVICES=0 python train.py -g -bs=16 -ne=0 -dd=./data/SNIPS -lod=./log/SNIPS -sd=./save/best/SNIPS -ld=./save/best/SNIPS -nh=8 -wed=64 -ied=64 -sdhd=64 -ln=SNIPS.txt 
