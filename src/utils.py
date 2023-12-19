import os 
import torch 
import matplotlib.pyplot as plt 

def save_model(epochs, model, optimizer, criterion, save_path):
    torch.save({
        'epoch': epochs, 
        'model_state_dict': model.state_dict(), 
        'optimizer_state_dict': optimizer.state_dict(), 
        'loss': criterion,
    }, save_path)

def load_model(model, model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def save_aleotoric_graph(lst, save_path):
    x = [i for i in range(1, len(lst)+1)]
    plt.figure(figsize=(8,7))
    plt.xlabel('Epochs')
    plt.ylabel('Losses in test set')
    plt.plot(x, lst, '.-', alpha=0.6)
    plt.savefig(save_path)

def save_epistemic_graph(lst, save_path):
    x = [i for i in range(1, len(lst)+1)]
    plt.figure(figsize=(8,7))
    plt.xlabel('Epochs')
    plt.ylabel('Losses in test set')
    plt.plot(x, lst, '.-', alpha=0.6)
    plt.savefig(save_path)