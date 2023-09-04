# Following Tutorial:
# https://jovian.ml/mailervivek/flower-classification/v/6
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
import torchvision
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# number of flower folders
OUTPUT_NODE = 331
BASE_DIR = "input/flowers-recognition/flowers/flowers"
MODEL_FILE_NAME = "flower-resnet.pth"
NUM_OF_EPOCHS = 30
BATCH_SIZE = 16

#NOTE: 
# if images that need to be classified should be put into 
# input\flowers-recognition\flowers\flowers\z_unclassified
# you will need need to know the length of images that need 
# to be classified
CLASSIFIED_IMAGES_COUNT = 2

#TODO: train the model for longer


def main():
    # get all sub folders from that directory
    os.listdir(BASE_DIR)
    loaded_model = ""
    
    try:
        loaded_model = torch.load(MODEL_FILE_NAME)
        print(f'model {MODEL_FILE_NAME} has been found')
    except:
        print(f"Could not find a model named {MODEL_FILE_NAME}. Starting a new Model")
        
    if(loaded_model):
        # use either CPU pf CUDA gpu
        device = get_default_device()

        # Augment Image
        transformer = torchvision.transforms.Compose(
        [   # Applying Augmentation
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomVerticalFlip(p=0.5),
            torchvision.transforms.RandomRotation(30),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])
        # load dataset, just folders of plants
        dataset = ImageFolder(BASE_DIR, transform=transformer)
        
        validation_size = 500
        training_size = len(dataset) - validation_size
        
        # Docs: https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split
        #                                   (Dataset, lenghts,                        denerator)   
        train_ds, val_ds_main = random_split(dataset, [training_size, validation_size])
        val_ds, test_ds = random_split(val_ds_main, [300, 200])
        # Docs: https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)
        # while loading datta use a specif device
        val_dl = DeviceDataLoader(val_dl, device)

        # make a model for loading
        model = ImageClassificationModel()
        # load the moddel
        model.load_state_dict(torch.load(MODEL_FILE_NAME))
        
        # move the tensors to a specif device
        to_device(model, device)

        # NOTE: Might need to delete the to_device above
        # sane as above 
        model = to_device(model, device)

        evaluate(model, val_dl)    

        # start the image classification
        # just assume that the image to be classified are at the end
        #             (start at 1,  to 2+1)
        for x in range(1, CLASSIFIED_IMAGES_COUNT+1):
            #get the last index 
            index = int(len(test_ds) - x )
            img, label = test_ds[index]
            # show what the NN thinks the image is
            print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model, device, dataset))
        
    else:
        
        transformer = torchvision.transforms.Compose(
        [   # Applying Augmentation
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomVerticalFlip(p=0.5),
            torchvision.transforms.RandomRotation(30),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
        )
        # turn images into tensors
        dataset = ImageFolder(BASE_DIR, transform=transformer)
        
        # ahow iamge
        show_example(*dataset[2])

        validation_size = 500
        training_size = len(dataset) - validation_size

        # foler names are the classea
        dataset.classes
        # Docs: https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split
        #                                   (Dataset, lenghts,                        denerator)   
        train_ds, val_ds_main = random_split(dataset, [training_size, validation_size])
        val_ds, test_ds = random_split(val_ds_main, [300, 200])

        # get length 
        len(train_ds), len(val_ds)

        # show image
        show_example(*train_ds[1])

        # Docs: https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)
        test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

        # plot on a graph 
        show_batch(train_dl)

        # make a model
        model = ImageClassificationModel()

        # NOTE: see if this does something, seems like it does nothing
        # get images and labels 
        for images, labels in train_dl:
            out = model(images)
            break
        
        # get the device, CPU or CUDA GPU
        device = get_default_device()


        # while loading datta use a specif device
        train_dl = DeviceDataLoader(train_dl, device)
        val_dl = DeviceDataLoader(val_dl, device)
        to_device(model, device)

        # move the tensors to a specif device
        model = to_device(ImageClassificationModel(), device)


        evaluate(model, val_dl)
        
        # https://optimization.cbe.cornell.edu/index.php?title=Adam#:~:text=Adam.,RMSP%20which%20are%20explained%20below.
        opt_func = torch.optim.Adam
        lr = 0.001

        # train the model
        history = fit(NUM_OF_EPOCHS, lr, model, train_dl, val_dl, opt_func)

        #plot the changes 
        plot_accuracies(history)

        test_dl = DeviceDataLoader(test_dl, device)
        evaluate(model, test_dl)

        history = fit(NUM_OF_EPOCHS, lr, model, train_dl, val_dl, opt_func)
        evaluate(model, test_dl)

        plot_accuracies(history)

        # save the model
        weights_fname = 'flower-resnet.pth'
        torch.save(model.state_dict(), weights_fname)

        # start the image classification
        # just assume that the image to be classified are at the end
        #             (start at 1,  to 2+1)
        for x in range(1, CLASSIFIED_IMAGES_COUNT+1):
            # get the last index
            index = int(len(test_ds) - x )
            img, label = test_ds[index]
            plt.imshow(img.permute(1, 2, 0))
            # show what the NN thinks the image is
            print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model, device, dataset))
        
# end of main()

# show the image
def show_example(img, label):
    plt.imshow(img.permute(1, 2, 0))
# end of show_example()


# plot the image
def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break
# end of show_batch()


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
# end of accuracy()

# Context-manager that disabled gradient calculation.
# https://pytorch.org/docs/stable/generated/torch.no_grad.html
@torch.no_grad()
def evaluate(model, val_loader):
    # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval
    # Sets the module in evaluation mode.
    model.eval()
    
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)
# end of evaluate


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history
# end of fit()

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
        
    else:
        return torch.device('cpu')
# end of get get_default_device()

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
# end of to_device()

def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
# end of plot_accuracies()

def predict_image(img, model, device, dataset):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze( 0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return dataset.classes[preds[0].item()]
# end of predict_image()

class ImageClassificationModel(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4

            nn.Flatten(),
            nn.Linear(256*28*28, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, OUTPUT_NODE))

    def forward(self, xb):
        return self.network(xb)

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

# end of ImageClassificationModel Class


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
# end of DeviceDataLoader Class


import torchvision.models as models
class ImageClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, OUTPUT_NODE)
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
# end of ImageClassificationModel Class

if __name__ == "__main__": 
    main()