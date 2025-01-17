import argparse
from ctgan import TVAE
from ctgan import load_demo
from ctgan.synthesizers.tvae import Decoder
import torch
import os
import csv
from tqdm import tqdm
import time
from ctgan.data_transformer import DataTransformer


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--compress_dims', type=tuple, default=(128, 128))
    parser.add_argument('--decompress_dims', type=tuple, default=(128, 128))
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--half', type=bool, default=True)
    parser.add_argument('--tag', type=str, default="gpu")
    parser.add_argument('--repeat', type=int, default=100)
    return parser.parse_args()

def main():
    args = parsing_argument()

    real_data = load_demo()
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
                batch_size=args.batch_size,
                embedding_dim=args.embedding_dim,
                decompress_dims=args.decompress_dims,
                cuda=args.cuda)
    tvae.decoder = Decoder(tvae.embedding_dim, tvae.decompress_dims, data_dim=156).to(tvae._device)
    tvae.decoder.load_state_dict(torch.load(f'decoder_{args.embedding_dim}.pt'))
    tvae.transformer = DataTransformer()
    tvae.transformer.fit(real_data, discrete_columns)


    for i in tqdm(range(args.repeat), desc=f"Repeat (batch {args.batch_size})"):
        sampling_latency, decoder_latency, inverse_transform_latency = tvae.benchmark(1024000, half = args.half)
        
        csvfile = f"{args.tag}_{args.embedding_dim}_b{args.batch_size}.csv" 
        if not os.path.isfile(csvfile):    
            with open(csvfile, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Embedding_Dim", "Decompress_Dims", "Batch_size", "Sampling Latency (ms)", "Decoder Latency (ms)", "Inverse Transform Latency (ms)"])
                writer.writerow([args.embedding_dim, args.decompress_dims, args.batch_size, round(sampling_latency,3 ), round(decoder_latency, 3), round(inverse_transform_latency, 3)])
        else:
            with open(csvfile, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([args.embedding_dim, args.decompress_dims, args.batch_size, round(sampling_latency,3 ), round(decoder_latency, 3), round(inverse_transform_latency, 3)])
        time.sleep(1)

if __name__ == '__main__':
    main()
