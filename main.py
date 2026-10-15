import os
import torch
from dataloader import load_data
from trainer import Trainer
from model import Model
from params import parse_args
from utils import set_seed, print_args, setup_logger
from loguru import logger


def main(args):
    setup_logger(args)
    print_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.seed is not None:
        set_seed(args.seed)

    data = load_data(args.data_dir, args.dataset, args.device, args.batch_size)

    model = Model(data, args).to(args.device)

    trainer = Trainer(model, data, args)

    if args.load_checkpoint:
        trainer.evaluate()
    else:
        logger.info("Start training the model")
        trainer.train_model()


if __name__ == '__main__':
    args = parse_args()
    main(args)


# python main.py --dataset taobao

# python main.py --dataset tmall

# python main.py --dataset jdata  --alpha 0.5