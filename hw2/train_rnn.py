import argparse
import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
from utils import TextDataset
from RNN import RNN


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a RNN model on a language modeling task")

    # data i/o
    parser.add_argument("--train_file", type=str, default="wiki.train", help="Path to the text file containing train data")
    parser.add_argument("--validation_file", type=str, default="wiki.valid", help="Path to the text file containing validation data")
    parser.add_argument("--test_file", type=str, default="wiki.test", help="Path to the text file containing test data")
    
    # training 
    parser.add_argument("--seed", type=int, default=1, help="A seed for reproducible training.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for training.")
    parser.add_argument("--hidden_size", type=float, default=1e-3, help="Size of the RNN hidden state dimension.")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="Total number of epochs to train over.")

    # logging
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store logs manually.")
    parser.add_argument("--log_to_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default='cs497-hw2')
    parser.add_argument("--wandb_tag", type=str, default='RNN')

    args = parser.parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def main():
    # housekeeping
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    # logging
    if args.log_to_wandb:
        wandb.init(
            project=args.wandb_project,
            tags=[args.wandb_tag],
            config=args
        )
        
    # load data
    train_dataset = TextDataset(args.train_file)
    vocab_size = train_dataset.return_vocabulary_size()
    val_dataset = TextDataset(args.validation_file, batch_size=1, train_dataset=train_dataset)
    test_dataset = TextDataset(args.test_file, batch_size=1, train_dataset=train_dataset)

    # TODO: dataloaders
    
    # load model
    model = RNN(args.hidden_size, vocab_size, device='cuda:0')



if __name__ == '__main__':
    main()