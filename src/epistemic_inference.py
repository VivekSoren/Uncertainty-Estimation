import argparse 
import datetime 
from tqdm import tqdm  
import torch 
import torch.optim as optim 
import torch.nn as nn 
import numpy as np 

from utils import load_model, save_epistemic_graph 
from dataset import OriginalDataset, ReducedDataset, map_function 
from model import MLP 

# Construct an argument parser 
parser = argparse.ArgumentParser()
parser.add_argument(
    '-r', '--run', type=int, 
    help="Number of attempt/run to run epistemic uncertainty"
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

    # dataset_train, dataset_valid, dataset_classes = original_dataset.get_datasets()
    dataset_test, dataset_classes = original_dataset.get_test_datasets()
    dataset_test = ReducedDataset(dataset_test, map_function)

    reduced_dataset_classes = {'0': 'abnormal', '1': 'normal'}

    print(f'Number of images for inference: {len(dataset_test)}')
    print(f'Class Names: {reduced_dataset_classes}\n')

    # Get a Randomized sampled dataset
    sampled_dataset = dataset_test.get_random_samples(dataset_test, split_ratio=0.8)
    test_loader = original_dataset.get_test_dataloader(sampled_dataset)

    # Inference parameters
    n_samples = 500
    hidden_layers = [800, 800]
    lr = 0.1 

    # Define the network
    models = [MLP(hidden_layers, droprates=[0, 0], n_classes=len(reduced_dataset_classes)), 
              MLP(hidden_layers, droprates=[0, 0.2], n_classes=len(reduced_dataset_classes)), 
              MLP(hidden_layers, droprates=[0, 0.4], n_classes=len(reduced_dataset_classes))]
    
    criterion = nn.CrossEntropyLoss().cuda()

    # Store the losses for all the models
    test_loss = []

    begin = datetime.datetime.now()

    for idx, model in enumerate(models):
        model_path = f'./outputs/run_{run}/models/model_{idx}.pth'
        model = load_model(model, model_path)
        model = model.to(device)
        model.train()

        optimizer = optim.SGD(model.parameters(), lr=lr)

        # Store the losses
        model_test_loss = []

        # Start the inference per model
        for sample in range(n_samples):
            print(f'Sample {sample+1}')

            # Testing 
            sample_loss, sample_acc, sample_error = test(model, test_loader, criterion)

            # Store the losses
            model_test_loss.append(sample_loss)

            # Print the losses
            print(f'Sample loss: {sample_loss:.3f}')
            print('-'*50)
        
        test_loss.append(model_test_loss)

        # print(f'Mean of the loss value for model_{idx}: {np.mean(model_test_loss)}')
        # print(f'Std Dev of the loss value for model_{idx}: {np.std(model_test_loss)}')

        graph_save_path = f'./outputs/run_{run}/epistemic_graph_{idx}.png'
        save_epistemic_graph(model_test_loss, graph_save_path)

    end = datetime.datetime.now()
    print(f'Time Elapsed: {end-begin}')

    for idx, r in enumerate(test_loss):
        print(f'Mean of the loss value for model_{idx}: {np.mean(r):.4f}')      # 0.3439, 0.3538, 0.4849
        print(f'Std Dev of the loss value for model_{idx}: {np.std(r):.4f}')    # 0.0000, 0.0084, 0.0111
