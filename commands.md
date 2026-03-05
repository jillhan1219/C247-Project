python -m emg2qwerty.train \
    user="single_user" \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    trainer.strategy=ddp \
    trainer.precision=16 \
    batch_size=32 \       
    num_workers=32 \
    optimizer.lr=0.001 