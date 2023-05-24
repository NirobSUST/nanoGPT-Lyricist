pip3 install torch torchvision torchaudio
pip install numpy
pip install transformers
pip install datasets
pip install tiktoken
pip install wandb
pip install tqdm
python data/lyrics/prepare.py
python train.py config/train_lyrics.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=5000 --lr_decay_iters=5000 --dropout=0.0
python sample.py --out_dir=out-lyrics --device=cpu --start="Love"