import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
import medmnist
from medmnist import INFO
import numpy as np
# from sklearn.model_selection import train_test_split

# Required Constants
data_flag = "octmnist"
download=True
NUM_WORKERS = 4     # Number of processes for parallel data preparation

class OriginalDataset(Dataset):
    def __init__(self, dataflag='octmnist', BATCH_SIZE = 64):
        
        super(Dataset, self).__init__()
        self.data_flag = dataflag
        self.download = download
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_WORKERS = NUM_WORKERS

        info = INFO[self.data_flag]
        task = info['task']
        self.n_channels = info['n_channels']
        self.n_classes = len(info['label'])

        self.DataClass = getattr(medmnist, info['python_class'])

    # Training transforms
    def get_train_transform(self):
        train_transform = transforms.Compose([
            # transforms.Resize((img_size, img_sizeI)),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        return train_transform

    def get_val_transform(self):
        val_transform = transforms.Compose([
            # transforms.Resize((img_size, img_size)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        return val_transform
    
    def get_test_transform(self):
        test_transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
            # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            # transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5return ), 

            transforms.ToTensor(), 
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        return test_transform
    
    def get_datasets(self):
        '''
        dataset_train: tuple of shape(1, 14)
                       dataset_train[idx][0] is of shape (1, 28, 28)       
        '''
        dataset_train = self.DataClass(split='train', 
                                        transform=self.get_train_transform(), 
                                        download=self.download)
        dataset_val = self.DataClass(split='val', 
                                        transform=self.get_val_transform(), 
                                        download=self.download)

        return dataset_train, dataset_val, dataset_train.info['label'] 
    
    def get_test_datasets(self):
        dataset_test = self.DataClass(split="test", 
                                      transform=self.get_test_transform(), 
                                      download=self.download)
        return dataset_test, dataset_test.info['label']

    def get_dataloaders(self, dataset_train, dataset_val):
        train_loader = DataLoader(
            dataset_train, batch_size=self.BATCH_SIZE, 
            shuffle=True, num_workers=self.NUM_WORKERS
        )
        val_loader = DataLoader(
            dataset_val, batch_size=self.BATCH_SIZE, 
            shuffle=False, num_workers=self.NUM_WORKERS
        )
        return train_loader, val_loader

    def get_test_dataloader(self, dataset_test):
        test_loader = DataLoader(
            dataset_test, batch_size=self.BATCH_SIZE, 
            shuffle=False, num_workers=self.NUM_WORKERS
        )
        return test_loader
    '''
    def get_target_source_dataset(self):
        
        # dataset_train: tuple of shape(1, 14)
        #                dataset_train[idx][0] is of shape (1, 28, 28)       
        
        dataset_train = self.DataClass(split='train', 
                                        transform=self.get_train_transform(), 
                                        download=self.download)
        dataset_valid = self.DataClass(split='val', 
                                        transform=self.get_val_transform(), 
                                        download=self.download)
        train_len = len(dataset_train)
        valid_len = len(dataset_valid)

        split_sizes = [train_len // 2, train_len - train_len // 2]
        split_train = random_split(dataset_train, split_sizes)
        source_train, target_train = split_train[0], split_train[1]

        split_sizes = [valid_len // 2, valid_len - valid_len // 2]
        split_valid = random_split(dataset_valid, split_sizes)
        source_valid, target_valid = split_valid[0], split_valid[1]

        return source_train, target_train, source_valid, target_valid
    '''
    def get_domain_split(self, dataset):
        dataset_len = len(dataset)
        split_sizes = [dataset_len // 2, dataset_len - dataset_len // 2]
        split_dataset = random_split(dataset, split_sizes)
        source, target = split_dataset[0], split_dataset[1]

        return source, target

def map_function(label):
    if label < 3:
        return 0
    else:
        return 1    

class ReducedDataset(OriginalDataset):
    def __init__(self, dataset, map_function):
        super(OriginalDataset, self).__init__()
        self.original_dataset = dataset
        self.map_function = map_function

    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        image, label = self.original_dataset[idx]
        # print(label)
        reduced_label = self.map_function(label)
        # print(reduced_label)
        return image, reduced_label
    
    def get_random_samples(self, dataset, split_ratio=0.8):
        dataset_len = len(dataset)
        split_sizes = [int(dataset_len * split_ratio), dataset_len - int(dataset_len * split_ratio)]
        split_dataset = random_split(dataset, split_sizes)
        sampled_dataset = split_dataset[0]

        return sampled_dataset 