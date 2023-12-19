import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import datetime
import argparse
import os

from dataset import OriginalDataset, ReducedDataset, map_function
from model import MLP, ResNet50
from utils import save_model

device = ('cuda' if torch.cuda.is_available() else 'cpu')

# Construct an argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '-r', '--run', type=int, default=1,
    help="Number of attempt/run to train the network"
)
args = vars(parser.parse_args())

def train(model, train_loader, optimizer, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    counter = 0 
    for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
        counter += 1
        image, labels = data 
        # labels = F.one_hot(labels.view(-1), num_classes=len(torch.unique(labels))).view(labels.shape[0], -1)
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward Pass 
        output = model(image)
        # Calculate the loss 
        loss = criterion(output, labels)
        train_running_loss += loss.item()

        # Backpropagation
        loss.backward()
        # Update the weights
        optimizer.step()
    # Loss for the complete epochs 
    epoch_loss = train_running_loss / counter 
    return epoch_loss

def validate(model, valid_loader, criterion):
    print(f'Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        counter += 1
        image, labels = data 
        image = image.to(device)
        labels = labels.to(device)
        # Forward pass
        output = model(image)
        # Calculate the loss 
        loss = criterion(output, labels)
        valid_running_loss += loss.item()
        # print(output, type(output))
        # print(labels, type(labels))
        
        # Calculate the accuracy 
        preds = torch.argmax(output.data, 1)
        valid_running_correct += (preds == labels).sum().item()

    epoch_acc = valid_running_correct / len(valid_loader.dataset)
    epoch_error = int(len(valid_loader.dataset) * (1 - epoch_acc))
    epoch_loss = valid_running_loss / counter 
    return epoch_loss, epoch_acc, epoch_error 

if __name__ == '__main__':

    run = args['run']
    
    original_dataset = OriginalDataset(dataflag='octmnist', BATCH_SIZE=64)

    dataset_train, dataset_valid, dataset_classes = original_dataset.get_datasets()
    dataset_train = ReducedDataset(dataset_train, map_function)
    dataset_valid = ReducedDataset(dataset_valid, map_function)

    reduced_dataset_classes = {'0': 'abnormal', '1': 'normal'}

    # source_train, target_train = original_dataset.get_domain_split(dataset_train)
    # print(len(source_train))
    # print(len(target_train))
    # # print(len(source_valid))
    # print(len(target_valid))

    print(f"Number of training images: {len(dataset_train)}")
    print(f"Number of validation images: {len(dataset_valid)}")
    print(f"Class names: {reduced_dataset_classes}\n")

    train_loader, valid_loader = original_dataset.get_dataloaders(dataset_train, dataset_valid)
    
    # Learning Parameters 
    hidden_layers = [800, 800]
    epochs = 100
    lr = 0.1 
    batch_size = 128
    momentum = 0 
     
    # Define networks
    mlp1 = [MLP(hidden_layers, droprates=[0, 0], n_classes=len(reduced_dataset_classes)),
            MLP(hidden_layers, droprates=[0, 0.2], n_classes=len(reduced_dataset_classes)), 
            MLP(hidden_layers, droprates=[0, 0.4], n_classes=len(reduced_dataset_classes)),
            ]
    # models = [ResNet50(n_classes=len(reduced_dataset_classes))]

    # Loss function
    criterion = nn.CrossEntropyLoss().cuda()

    # Create the output directory
    output_path = f'./outputs/run_{run}'
    os.makedirs(output_path, exist_ok=True)
    model_save_dir = f'{output_path}/models'
    os.makedirs(model_save_dir, exist_ok=True)

    begin = datetime.datetime.now()

    # Start the training
    for idx, mlp in enumerate(mlp1):
        mlp = mlp.to(device)
        # Optimizer 
        optimizer = optim.SGD(mlp.parameters(), lr=lr)

        train_loss, valid_loss = [], []
        
        for epoch in range(epochs):
            print(f'Epoch {epoch+1} for {epochs}')
            
            # Train 
            train_epoch_loss = train(mlp, train_loader, optimizer, criterion)
            valid_epoch_loss, valid_epoch_acc, valid_epoch_error = validate(mlp, valid_loader, criterion)

            # Store the losses
            train_loss.append(train_epoch_loss)
            valid_loss.append(valid_epoch_loss)
            # valid_error.append(valid_epoch_error)

            # Print the losses
            print(f'Training loss: {train_epoch_loss:.3f}')
            print(f'Valid loss: {valid_epoch_loss:.3f}')
            
            print('-'*50)

        # Save the torch models
        model_save_path = f'{model_save_dir}/model_{str(idx)}.pth'
        save_model(epochs, mlp, optimizer, criterion, model_save_path)

    end = datetime.datetime.now()
    print(f'Time elapsed: {end - begin}')