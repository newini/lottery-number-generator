#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =================================================
# generator.py: Expected number generator for Lottery game
__author__ = "Eunchong Kim"
__copyright__ = "Copyright 2021, The Lottery expected number generator Project"
__credits__ = ["Eunchong Kim"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "Eunchong Kim"
__email__ = "chariskimec@gmail.com"
__status__ = "Dev"


# =================================================
# modules
import argparse, configparser, copy, csv, datetime, logging, random # default modules
import coloredlogs
import numpy as np
import torch, torchsummary
from sklearn.model_selection import train_test_split


# =================================================
# Parse arguments
parser = argparse.ArgumentParser()

parser.add_argument("-c", "--config_file", type=str, help="Specify config file")

parser.add_argument('-d', '--debug', action='store_true', help='Debug mode')
parser.add_argument('-l', '--log_file_name', type=str, default=datetime.date.today().strftime('%Y%m')+'.log',
        help='Log file name')

parser.add_argument('-L', '--Lottery_max_number', type=int, default=31,
        help='Lottery game max numbers. Default: 31')
parser.add_argument('-p', '--pick', type=int, default=5,
        help='Number of pick in one game. Default: 5')

parser.add_argument('-f', '--file_path', type=str, help='CSV file path')
parser.add_argument('-r', '--remove_lines', type=int, default=0,
        help='Remove unnecessary header lines in csv file. Default: 0')
parser.add_argument('-a', '--appearance_first_number_order', type=int, default=0,
        help='Order of First number appears on the row in csv file. Default: 0')

parser.add_argument('--random_seed', type=int, default=7, help='Random seed. Default: 7')

#parser.add_argument('-e', '--epochs', type=int, default=50)
#parser.add_argument('-t', '--times', type=int, default=5)
#parser.add_argument('-b', '--batch_size', type=int, default=128)
#parser.add_argument('-c', '--check', action='store_true')
#parser.add_argument('-p', '--predict_drawing_n', type=int, default=5)
args = parser.parse_args()

# Read from config file and overwrite
if args.config_file:
    config = configparser.ConfigParser()
    config.read(args.config_file)
    defaults = {}
    defaults.update(dict(config.items("Defaults")))
    parser.set_defaults(**defaults)
    args = parser.parse_args()


# =================================================
# Logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="log/%s" % (args.log_file_name),
    filemode="a",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)-8s %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

# color logging
coloredlogs.install()

logging.info('Logging setup finished')


# =================================================
# Setup
logging.info('Debug mode: ' + str(args.debug))
# Fix random seed
random.seed(args.random_seed)


# =================================================
# Read lottery number csv
if not args.file_path:
    logging.error('No csv file path given!')
    exit(1)

A_n = [] # pick_in_decimal_w_times = []
B_n = [] # pick_in_binary_w_times = []
C_n = [] # pick_in_binary_accumulated_w_times = []
P_np1 = []
with open(args.file_path) as csvfile:
    # Remove unnecessary lines
    for i in range(args.remove_lines):
        next(csvfile)

    C_k = np.array([0.0 for i in range(args.Lottery_max_number)])

    csvreader = csv.reader(csvfile, delimiter=',')
    for index, row in enumerate(csvreader):
        A_k = []
        B_k = [ 0.0 for i in range(args.Lottery_max_number) ]
        for num in row[args.appearance_first_number_order:args.appearance_first_number_order+args.pick]:
            A_k.append(int(num)-1) ### Becareful
            B_k[int(num)-1] = 1
        C_k += np.array(B_k)

        A_n.append(A_k)
        B_n.append(B_k)
        C_n.append(copy.copy(C_k)) # np.array
        R_k = C_k/( args.pick*(index+1) )
        P_np1.append( (1 - R_k)/(args.Lottery_max_number-1) )


logging.info('Total number of times: %d' % len(A_n))

# Check probability = 1 each games
for index, P_kp1 in enumerate(P_np1):
    if round( np.sum(P_kp1), 9 ) != 1:
        logging.warning('k=%d, Probability is not 1' % index)





