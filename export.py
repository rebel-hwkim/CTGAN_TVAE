import argparse
import os

from ctgan import TVAE
from ctgan import load_demo
from ctgan.synthesizers.tvae import Decoder
import torch
import torch.nn as nn


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--compress_dims', type=tuple, default=(128, 128))
    parser.add_argument('--decompress_dims', type=tuple, default=(128, 128))
    parser.add_argument('--cuda', type=bool, default=True)
    return parser.parse_args()


def main():
    args = parsing_argument()

    real_data = load_demo()

    # Names of the columns that are discrete
    discrete_columns = [
        'workclass',
        'education',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native-country',
        'income'
    ]

    tvae = TVAE(epochs=10,
                batch_size=512,
                embedding_dim=args.embedding_dim,
                compress_dims=args.compress_dims,
                decompress_dims=args.decompress_dims,
                cuda=args.cuda,
                verbose=True)
    tvae.fit(real_data, discrete_columns)

    # Create synthetic data
    synthetic_data = tvae.sample(100)
    print(synthetic_data)
    
    torch.save(tvae.decoder.state_dict(), f'decoder_{args.embedding_dim}.pt')

if __name__ == '__main__':
    main()