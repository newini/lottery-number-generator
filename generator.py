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
# Config
parser.add_argument("-c", "--config_file", type=str, help='Config file for the lottery type (default: nil)')
# Debug and log
parser.add_argument('-d', '--debug', action='store_true', help='Debug mode (default: false)')
parser.add_argument('-l', '--log_file_name', type=str, default=datetime.date.today().strftime('%Y%m')+'.log',
        help='Log file name (default: %Y%m.log)')
# Lottery game definition
parser.add_argument('-L', '--Lottery_max_number', type=int, default=31,
        help='Lottery game max numbers. (default: 31)')
parser.add_argument('-p', '--pick', type=int, default=5,
        help='Number of pick in one game (default: 5)')
# Detail of input data
parser.add_argument('-f', '--file_path', type=str, help='CSV file path (default: nil)')
parser.add_argument('-r', '--remove_lines', type=int, default=0,
        help='Remove unnecessary header lines in csv file (default: 0)')
parser.add_argument('-a', '--appearance_first_number_order', type=int, default=0,
        help='Order of First number appears on the row in csv file (default: 0)')
parser.add_argument('--perge_data_percentage', type=float, default=0.0,
        help='Perge older data to improve accuracy. 0.0 < Input in percentage < 1.0 (default: 0.0)')
# For formating data
parser.add_argument('--test_size', type=float, default=0.1,
        help='Test size is for validation. 0.0 < Input in percentage < 1.0 (default: 0.1)')
parser.add_argument('--random_state', type=int, default=0,
        help='Random state is used when split data for validation (default: 0)')
parser.add_argument('--batch_size', type=int, default=1,
        help='How many samples per batch to load (default: 1)')
# For training
parser.add_argument('-e', '--epochs', type=int, default=50,
        help='How many times to train (default: 50)')

# Others
parser.add_argument('--random_seed', type=int, default=7, help='Random seed. Default: 7')

#parser.add_argument('-t', '--times', type=int, default=5)
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
    args = parser.parse_args() # Overwrite arguments


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

N = len(A_n)

logging.info('Total number of times: %d' % N)

# Check probability = 1 each games
for index, P_kp1 in enumerate(P_np1):
    if round( np.sum(P_kp1), 9 ) != 1:
        logging.warning('k=%d, Probability is not 1' % index)


# =================================================
# Format data
X_array = []
y_array = []

# Simplest model
# L channel input --> L channel output
for index in range( int(N*args.perge_data_percentage), N-1 -1 ): # Do not use last data for training
    X_array.append( B_n[index] )
    y_array.append( A_n[index+1] )

logging.info( 'Lengh of X is %d' % ( len(X_array) ) )

# List --> numpy.array
X_array = np.array(X_array)
y_array = np.array(y_array)

# Split for valid
X_train, X_valid, y_train, y_valid = train_test_split(
        X_array, y_array, test_size=args.test_size, random_state=args.random_state)

# numpy.array --> torch.tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.long)
logging.info('X_train size is: %s' % str( X_train_tensor.shape ) )
logging.info('y_train size is: %s' % str( y_train_tensor.shape ) )
logging.info('X_valid size is: %s' % str( X_valid_tensor.shape ) )
logging.info('y_valid size is: %s' % str( y_valid_tensor.shape ) )

# Create dataset
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
valid_dataset = torch.utils.data.TensorDataset(X_valid_tensor, y_valid_tensor)

# Create data loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size)


# =================================================
# Train model function
def trainModel(model, loss_function, optimizer, data_loader, is_validation=False):
    if is_validation:
        model.eval()

    loss_error = 0
    correct_count = 0
    count = len(data_loader.dataset)
    if args.debug:
        print(count)

    for X, y in data_loader:
        # Get predicted result
        predicted_numbers_binary = model(X)

        # Get top pick numbers
        _, predicted_numbers = torch.topk(predicted_numbers_binary.data, args.pick)

        if args.debug:
            print(X.shape)
            print(y.shape)
            print(predicted_numbers_binary.shape)
            print(predicted_numbers.shape)

        # Check if predicted contains its next numbers, and count
        corrects = torch.eq(predicted_numbers, y)
        #for i in range(args.predict_drawing_n):
        #    if i == 0:
        #        corrects = torch.eq(all_number_in6_tensor[y+1], predicted_numbers[:,i].reshape(-1, 1))
        #    else:
        #        corrects += torch.eq(all_number_in6_tensor[y+1], predicted_numbers[:,i].reshape(-1, 1))
        correct_count += corrects.sum().item()/args.pick

        if args.debug:
            print(correct_count)

        # Set one target
        # Loss function only accept 1D target
        #targets = torch.tensor( [random.choices([i for i in range(total_numbers)], weights=(1-all_number_probability_in45[yy+1]), k=1) for yy in y] ).reshape(-1)
        targets = torch.tensor( [random.choice(y_n) for y_n in y] ).reshape(-1)

        # Calculate loss error
        loss = loss_function(predicted_numbers_binary, targets)
        loss_error += loss.item()*len(y)

        # Update weight
        if not is_validation:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if args.debug:
            break

    mean_loss_error = loss_error / count
    accuracy = correct_count / count

    if is_validation:
        model.train()
    return mean_loss_error, accuracy


# =================================================
# Execute
def execute(model, loss_function, optimizer, train_loader, valid_loader, epoch):
    max_epoch = 0
    max_model = model
    max_accuracy = 0.0
    for i in range(epoch):
        train_loss, train_accuracy = trainModel(model, loss_function, optimizer, train_loader)
        valid_loss, valid_accuracy = trainModel(model, loss_function, optimizer, valid_loader, is_validation=True)

        logging.info('Epoch: {:04d}, Train loss: {:.4f}, acc.: {:.4f}, Valid loss: {:.4f}, acc.: {:.4f}'.format(i, train_loss, train_accuracy, valid_loss, valid_accuracy))

        if i > 10 and valid_accuracy > max_accuracy:
            logging.info('Found max accuracy model. valid_accuracy: %f' % valid_accuracy)
            max_accuracy = valid_accuracy
            max_model = copy.deepcopy(model)
            max_epoch = i
        if args.debug:
            break

    logging.info('In epoch %d, max_accuracy is %f' % (max_epoch, max_accuracy))
    return model, max_model


# Set model
torch.manual_seed(0)
model = torch.nn.Sequential(
        torch.nn.Linear(args.Lottery_max_number, args.Lottery_max_number*2),
        torch.nn.Linear(args.Lottery_max_number*2, args.Lottery_max_number),
        #torch.nn.ReLU(),
        #torch.nn.Dropout(p=0.5),
        )
logging.info('Model is: %s' % str( torchsummary.summary(model, (args.Lottery_max_number,)) ) )

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

trained_model, max_trained_model = execute(model, loss_function, optimizer, train_loader, valid_loader, args.epochs)


# =================================================
# Test
def testModel(model):
    logging.info('[Test]')
    model.eval()
    for n in range(N-1 -5, N-1):
        X = B_n[n]

        predicted_numbers_binary = model( torch.tensor(X, dtype=torch.float) )
        _, predicted_numbers = torch.topk(predicted_numbers_binary.data, args.pick)

        # Go back to original decimal
        #predicted_numbers = predicted_numbers.sort()[0] + 1
        predicted_numbers = predicted_numbers + 1
        answer_numbers = np.array(A_n[n+1]) + 1
        logging.info('n=%d, A`_{n+1}=%s. A_{n+1}=%s'
                % (n+1, str(predicted_numbers.tolist()), str(answer_numbers.tolist()) ) )

logging.info('Trained model:')
testModel(trained_model)
logging.info('Max Trained model:')
testModel(max_trained_model)
