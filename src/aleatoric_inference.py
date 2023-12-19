import argparse
import datetime
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from utils import load_model, save_aleotoric_graph
from dataset import OriginalDataset, ReducedDataset, map_function
from model import MLP

# Construct an arguemnt parser 
parser = argparse.ArgumentParser()
parser.add_argument(
    '-r', '--run', type=int, 
    help="Number of attempt/run to run aleotoric uncertainty"
)
args = vars(parser.parse_args())

device = ('cuda' if torch.cuda.is_available() else 'cpu')

def test(model, test_loader, criterion):
    print(f'Testing')
    test_running_loss = 0.0
    test_running_correct = 0 
    counter = 0
    for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        # Forward pass
        output = model(image)
        # Calculate the loss 
        loss = criterion(output, labels)
        test_running_loss += loss.item()
        # Calculate the accuracy
        preds = torch.argmax(output.data, 1)
        test_running_correct += (preds == labels).sum().item()

    epoch_acc = test_running_correct / len(test_loader.dataset)
    epoch_error = int(len(test_loader.dataset) * (1 - epoch_acc))
    epoch_loss = test_running_loss / counter 

    return epoch_loss, epoch_acc, epoch_error

if __name__ == '__main__':

    run = args['run']

    original_dataset = OriginalDataset(dataflag='octmnist', BATCH_SIZE=64)

    dataset_train, dataset_valid, dataset_classes = original_dataset.get_datasets()
    dataset_test, dataset_classes = original_dataset.get_test_datasets()
    dataset_test = ReducedDataset(dataset_test, map_function)
    
    reduced_dataset_classes = {'0': 'abnormal', '1': 'normal'}

    print(f"Number of images for inference: {len(dataset_test)}")
    print(f'Class names: {reduced_dataset_classes}\n')

    # Inference parameters
    n_samples = 500
    hidden_layers= [800, 800]
    lr = 0.1

    # Define the network
    model = MLP(hidden_layers, droprates=[0, 0], n_classes=len(reduced_dataset_classes))
    model_path = f'./outputs/run_{run}/models/model_0.pth'
    model = load_model(model, model_path)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    begin = datetime.datetime.now()

    # Store the losses
    test_loss = []

    # Start the inference
    for sample in range(n_samples):
        print(f'Sample {sample}')

        # Create a new randomized dataset 
        sampled_dataset = dataset_test.get_random_samples(dataset_test, split_ratio=0.8)

        # Create a dataloader
        test_loader = original_dataset.get_test_dataloader(sampled_dataset)

        # Testing
        sample_loss, sample_acc, sample_error = test(model, test_loader, criterion)

        # Store the losses
        test_loss.append(sample_loss)

        # Print the losses
        print(f'Sample loss: {sample_loss:.3f}')
        print('-'*50)

    end = datetime.datetime.now()
    print(f'Time Elapsed: {end-begin}')
    # print(test_loss)

    graph_save_path = f'./outputs/run_{run}/aleotoric_graph.png'
    save_aleotoric_graph(test_loss, graph_save_path)

    print(f'Mean of the loss value: {np.mean(test_loss)}')      # 0.3699
    print(f'Std Dev of the test loss: {np.std(test_loss)}')     # 0.0102
    
