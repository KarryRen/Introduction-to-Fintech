import os

import sys

sys.path.append('/Users/lanyang/Desktop/Machine-Learning-in-the-Chinese-Stock-Market-Reproduction-main/lanyang/utils')

print(sys.path)
from format_transfer import mon_freq_data

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path',default='.')
args = parser.parse_args()

mon_freq_data(tmp,tmp.columns[2:],args.path)