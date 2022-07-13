import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

MNIST_MAX_VAL = 255.0

def create_data_loaders(df_path=None, train_batch_size=1000, test_batch_size=1000, test_portion=0.2, df=None):
    """Creates data loaders for train and test"""
    if df is None:
        df = pd.read_csv(df_path)
    labels = df["label"].values
    images = (df.iloc[:, 1:].values).astype("float32")
    train_images, val_images, train_labels, val_labels = train_test_split(images, labels,
        stratify=labels, test_size = test_portion)

    train_images = train_images.reshape(train_images.shape[0], 1, 28, 28)
    val_images = val_images.reshape(val_images.shape[0], 1, 28, 28)

    #train
    train_images_tensor = torch.tensor(train_images)/MNIST_MAX_VAL
    train_labels_tensor = torch.tensor(train_labels)
    train_tensor = TensorDataset(train_images_tensor, train_labels_tensor)

    #val
    val_images_tensor = torch.tensor(val_images)/MNIST_MAX_VAL
    val_labels_tensor = torch.tensor(val_labels)
    val_tensor = TensorDataset(val_images_tensor, val_labels_tensor)

    train_loader = DataLoader(train_tensor, batch_size=train_batch_size, num_workers=2, shuffle=True)
    val_loader = DataLoader(val_tensor, batch_size=test_batch_size, num_workers=2, shuffle=True)
    
    return train_loader, val_loader