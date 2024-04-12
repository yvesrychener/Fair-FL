import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--home')
    args = parser.parse_args()
    HOMEFOLDER = args.home
    os.mkdir(os.path.join(HOMEFOLDER, 'datasets'))
    os.mkdir(os.path.join(HOMEFOLDER, 'results'))
    os.mkdir(os.path.join(HOMEFOLDER, 'results/agnostic'))
    os.mkdir(os.path.join(HOMEFOLDER, 'results/clientwise'))
    os.mkdir(os.path.join(HOMEFOLDER, 'results/ours'))
