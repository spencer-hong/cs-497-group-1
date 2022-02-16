import argparse
import os
import random
import math

from tqdm.auto import tqdm
import wandb

import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader

from utils import TextDataset, text_collate_fn, compute_perplexity, save_perplexity
from RNN import RNN


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a RNN model on a language modeling task")

    # data i/o
    parser.add_argument("--train_file", type=str, default="wiki.train.txt", help="Path to the text file containing train data")
    parser.add_argument("--validation_file", type=str, default="wiki.valid.txt", help="Path to the text file containing validation data")
    parser.add_argument("--test_file", type=str, default="wiki.test.txt", help="Path to the text file containing test data")
    
    # training 
    parser.add_argument("--seed", type=int, default=1, help="A seed for reproducible training.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for training.")
    parser.add_argument("--hidden_size", type=int, default=100, help="Size of the RNN hidden state dimension.")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="Total number of epochs to train over.")
    parser.add_argument("--batch_size", type=int, default=20, help="Size of train batches.")
    parser.add_argument("--device", type=str, default='cuda:0', help="Device to use for training/eval.")

    # logging
    parser.add_argument("--output_dir", type=str, default='rnn_sweep', help="Where to store logs manually.")
    parser.add_argument("--log_step_interval", type=int, default=10, help="Log perplexity every _log_step_interval_ training steps.")
    parser.add_argument("--log_to_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default='cs497-hw2')
    parser.add_argument("--wandb_tag", type=str, default='RNN')

    args = parser.parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


debugging = False
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
        wandb.run.name = 'hidden size: ' + str(args.hidden_size) + '; learning rate: ' + str(args.learning_rate)
        
    # load data
    train_dataset = TextDataset(args.train_file, debugging=debugging)
    vocab_size = train_dataset.return_vocabulary_size()
    val_dataset = TextDataset(args.validation_file, batch_size=1, train_dataset=train_dataset, debugging=debugging)
    test_dataset = TextDataset(args.test_file, batch_size=1, train_dataset=train_dataset, debugging=debugging)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=text_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=text_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=text_collate_fn)

    # load model; training-related logistics
    model = RNN(args.hidden_size, vocab_size, device=args.device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    loss_func = CrossEntropyLoss()

    # logging initialization
    train_perplexity, val_perplexity, test_perplexity = [], [], []

    def evaluate_model(dataloader):
        avg_ppl = 0
        prev_hidden = None
        for step, (source, target) in enumerate(dataloader):
            source, target = source.to(args.device), target.to(args.device)
            outputs = model(source, prev_hidden)
            probabilities, prev_hidden = outputs['probabilities'], outputs['hidden_state']
            loss = loss_func(probabilities, target)
            ppl = compute_perplexity(loss)
            avg_ppl += ppl
        avg_ppl /= len(dataloader)
        return avg_ppl

    # progress bar init
    num_train_steps = len(train_dataloader) * args.num_train_epochs
    progress_bar = tqdm(range(num_train_steps))

    for epoch in range(args.num_train_epochs):
        model.train()
        prev_hidden = None
        for step, (source, target) in enumerate(train_dataloader):
            source, target = source.to(args.device), target.to(args.device)
            outputs = model(source, prev_hidden)
            probabilities, prev_hidden = outputs['probabilities'], outputs['hidden_state']
            
            # compute loss and perplexity
            loss = loss_func(probabilities, target)
            ppl = compute_perplexity(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # progress bar / log
            progress_bar.update(1)
            progress_bar.set_description("Train perplexity: %s" % ppl)
            if args.log_to_wandb:
                wandb.log({"train/perplexity": ppl})
            train_perplexity.append(ppl)
        
        model.eval()
        avg_val_ppl = evaluate_model(val_dataloader)
        avg_test_ppl = evaluate_model(test_dataloader)

        val_perplexity.append(avg_val_ppl)
        test_perplexity.append(avg_test_ppl)
        if args.log_to_wandb:
            wandb.log({
                "validation/perplexity": avg_val_ppl,
                "test/perplexity": avg_test_ppl
            })
      
    if args.output_dir is not None:  
        save_perplexity(train_perplexity, args.output_dir, 'train')
        save_perplexity(val_perplexity, args.output_dir, 'val')
        save_perplexity(test_perplexity, args.output_dir, 'test')
    print('done!')


if __name__ == '__main__':
    main()