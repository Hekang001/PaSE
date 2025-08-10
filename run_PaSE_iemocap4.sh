# IEMOCAPFour
python -u MoMKE/train.py --dataset=IEMOCAPFour --audio-feature=wav2vec-large-c-UTT --text-feature=deberta-large-4-UTT --video-feature=manet_UTT --seed=66 --batch-size=64 --epoch=200 --lr=0.0001 --hidden=256 --depth=4 --num_heads=2 --drop_rate=0.5 --attn_drop_rate=0.0 --test_condition=atv --gpu=0
