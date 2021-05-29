import os
import sys
import argparse
import importlib

_SOURCE_PATH_ = os.path.abspath(__file__)
_SOURCE_DIRECTORY_ = os.path.dirname(_SOURCE_PATH_)

sys.path.append(_SOURCE_DIRECTORY_)

parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, required=True, help='choose one of network in gans directory')
args = parser.parse_args()

importlib.import_module('gans.' + args.net)