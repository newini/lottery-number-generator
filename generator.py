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
# Import modules
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
def setupLogging():
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
# Read lottery number csv
A_n = [] # pick in decimal w times
B_n = [] # pick in binary w times
C_n = [] # pick in binary accumulated w times
P_np1 = []
def readCSV():
    if not args.file_path:
        logging.error('No csv file path given!')
        exit(1)

    with open(args.file_path) as csvfile:
        # Remove unnecessary lines
        for i in range(args.remove_lines):
            next(csvfile)

        C_k = np.array( [0.0 for i in range(args.Lottery_max_number)] )

        csvreader = csv.reader(csvfile, delimiter=',')
        for index, row in enumerate(csvreader):
            A_k = []
            B_k = [ 0.0 for i in range(args.Lottery_max_number) ]
            for num in row[args.appearance_first_number_order:args.appearance_first_number_order+args.pick]:
                A_k.append(int(num)-1)      # Becareful!
                B_k[int(num)-1] = 1         # minus 1 to adapt to array index
            C_k += np.array(B_k)

            A_n.append(A_k)
            B_n.append(B_k)
            C_n.append(copy.copy(C_k)) # np.array
            R_k = C_k/( args.pick*(index+1) )
            P_np1.append( (1 - R_k)/(args.Lottery_max_number-1) )

    global N
    N = len(A_n)

    logging.info('N = %d' % N)

    # Check probability = 1 each games
    for index, P_kp1 in enumerate(P_np1):
        if round( np.sum(P_kp1), 9 ) != 1:
            logging.warning('k=%d, Probability is not 1' % index)


# =================================================
# Format data
def formatData(X_array, y_array):
    # List --> numpy.array
    X_array = np.array(X_array)
    y_array = np.array(y_array)

    # Split for valid
    X_train, X_valid, y_train, y_valid = train_test_split(
            X_array, y_array, test_size=args.test_size, random_state=args.random_state)

    # numpy.array --> torch.tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float)
    X_valid_tensor = torch.tensor(X_valid, dtype=torch.float)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.float)
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

    return train_loader, valid_loader


# =================================================
# Train model function
def trainModel(model, loss_function, optimizer, data_loader, is_validation=False, k=1):
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
        _, predicted_numbers = torch.topk(predicted_numbers_binary.data, k)

        _, y_topk = torch.topk(y, k)

        # Check if predicted contains its next numbers, and count
        corrects = torch.eq( predicted_numbers.sort()[0], y_topk.sort()[0] )
        correct_count += corrects.sum().item()/args.pick

        if args.debug:
            print(X.shape)
            print(y.shape)
            print(predicted_numbers_binary.shape)
            print(predicted_numbers.shape)
            print(correct_count)

        # Calculate loss error
        loss = loss_function(predicted_numbers_binary, y)
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
# Initialize
def initialize():
    setupLogging()

    logging.info('Debug mode: ' + str(args.debug))

    # Fix random seed
    random.seed(args.random_seed)

    torch.manual_seed(0)


# =================================================
# Execute
def execute(model, loss_function, optimizer, train_loader, valid_loader, k=1):
    max_epoch = 0
    max_model = model
    max_accuracy = 0.0
    for i in range(args.epochs):
        train_loss, train_accuracy = trainModel(model, loss_function, optimizer, train_loader, k=k)
        valid_loss, valid_accuracy = trainModel(model, loss_function, optimizer, valid_loader, is_validation=True, k=k)

        logging.info('e={:04d}, Train loss={:.4f}, acc={:.4f}. Valid loss={:.4f}, acc.={:.4f}'.format(i, train_loss, train_accuracy, valid_loss, valid_accuracy))

        if i > 10 and valid_accuracy > max_accuracy:
            logging.info('Found max accuracy model. valid_accuracy: %f' % valid_accuracy)
            max_accuracy = valid_accuracy
            max_model = copy.deepcopy(model)
            max_epoch = i
        if args.debug:
            break

    logging.info('In epoch %d, max_accuracy is %f' % (max_epoch, max_accuracy))
    return model, max_model


# =================================================
def finalize():
    pass



# =================================================
# Test
def testModel(model, X, k=1):
    model.eval()
    predicted_numbers_binary = model( torch.tensor(X, dtype=torch.float) )
    _, predicted_numbers = torch.topk(predicted_numbers_binary.data, k)

    # Go back to original decimal
    predicted_numbers = predicted_numbers + 1
    return predicted_numbers


# =================================================
# Simplest model
# model: L channel input --> L channel output --> get top 5 numbers
# max train acc=0.13, max valid acc=0.05
def simplestModelExec():
    X_array = []
    y_array = []

    print(N)
    for index in range( int(N*args.perge_data_percentage), N-1 -1 ): # Do not use last data for training
        X_array.append( B_n[index] + P_np1[index].tolist() )
        y_array.append( B_n[index+1] )
    logging.info( 'Lengh of X is %d' % ( len(X_array) ) )

    train_loader, valid_loader = formatData(X_array, y_array)

    model = torch.nn.Sequential(
            torch.nn.Linear(args.Lottery_max_number*2, args.Lottery_max_number),
            #torch.nn.ReLU(),
            #torch.nn.Dropout(p=0.5),
            )
    logging.info('Model is: %s' % str( torchsummary.summary(model, (args.Lottery_max_number*2,)) ) )

    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    trained_model, max_trained_model = execute(model, loss_function, optimizer, train_loader, valid_loader, k=args.pick)

    logging.info('Test w/ trained model')

    for n in range(N-6, N):
        X = B_n[n] + P_np1[n].tolist()
        predicted_numbers = testModel(trained_model, X, k=args.pick)
        if n < N-1:
            answer_numbers = np.array(A_n[n+1]) + 1
        else:
            answer_numbers = np.array([])
        logging.info('n=%d, A`_{n+1}=%s. A_{n+1}=%s'
                % (n+1, str(predicted_numbers.tolist()), str(answer_numbers.tolist()) ) )

    #logging.info('Test w/ Max valid accuracy model')

# Each pick model
# X = [ A_n[0:5] ], y = A_{n+1}[0][ picks (n) ], [ picks (n+1) ]
def eachPickModelExec():
    X_array = []
    y_array = []

    for index in range( int(N*args.perge_data_percentage), N-1 -1 ): # Do not use last data for training
        for i in range(args.pick):
            X_array.append( B_n[index] )
            y_array.append( A_n[index+1] )

    logging.info( 'Lengh of X is %d' % ( len(X_array) ) )

    train_loader, valid_loader = formatData(X_array, y_array)

    model = torch.nn.Sequential(
            torch.nn.Linear(args.Lottery_max_number, args.Lottery_max_number),
            #torch.nn.ReLU(),
            #torch.nn.Dropout(p=0.5),
            )
    logging.info('Model is: %s' % str( torchsummary.summary(model, (args.Lottery_max_number,)) ) )


    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    logging.info('Test w/ trained model')
    testModel(trained_model, A_n, B_n, N)
    logging.info('Test w/ Max valid accuracy model')
    testModel(max_trained_model, A_n, B_n, N)


# =================================================
# Main
if __name__ == "__main__":
    initialize()

    readCSV()

    simplestModelExec()
