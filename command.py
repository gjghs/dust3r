import os
import click
from argparse import ArgumentParser

# ROCR_VISIBLE_DEVICES for specify ROCm device

commands = {
    1: 'CUDA_VISIBLE_DEVICES=7 python demo.py --weights checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth',
    2: 'CUDA_VISIBLE_DEVICES=7 python inference.py'
    }

parser = ArgumentParser()
parser.add_argument('index', type=int)
args = parser.parse_args()

def main():
    os.system(commands[args.index])

if __name__ == '__main__':
    main()