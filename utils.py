import torch
from torch import nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.dataset import Dataset, random_split
import os
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
from wandb import Api

animal_names = [
    "Aardvark", "Albatross", "Anteater", "Armadillo",
    "Baboon", "Beaver", "Bison", "Buffalo",
    "Camel", "Cheetah", "Cougar", "Chinchilla",
    "Dingo", "Dolphin", "Duck", "Deer",
    "Elephant", "Echidna", "Emu", "Eagle",
    "Ferret", "Flamingo", "Fox", "Falcon",
    "Giraffe", "Gazelle", "Gecko", "Gorilla",
    "Hedgehog", "Hamster", "Hummingbird", "Heron",
    "Iguana", "Impala", "Ibis", "Inchworm",
    "Jackal", "Jaguar", "Jackrabbit", "Jellyfish",
    "Kangaroo", "Koala", "Kudu", "Kinkajou",
    "Lemur", "Lynx", "Llama", "Leopard",
    "Meerkat", "Mongoose", "Mole", "Manatee",
    "Narwhal", "Numbat", "Newt", "Nightingale",
    "Ocelot", "Ostrich", "Orangutan", "Octopus",
    "Penguin", "Platypus", "Porcupine", "Puffin",
    "Quokka", "Quail", "Quetzal", "Quoll",
    "Raccoon", "Rabbit", "Raven", "Rhinoceros",
    "Salamander", "Squirrel", "Seal", "Swan",
    "Tapir", "Toucan", "Tortoise", "Turtle",
    "Umbrellabird", "Urial", "Uakari", "Urchin",
    "Vulture", "Viper", "Vicuna", "Vicuña",
    "Wallaby", "Wombat", "Walrus", "Woodpecker",
    "Xerus", "Xenops", "Xantus", "Xiphias",
    "Yak", "Yabby", "Yellowjacket", "Yakutian",
    "Zebra", "Zebu", "Zorilla"
]

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = np.argmax(labels, axis=1).flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def animal_version_name(project_name):
    api = Api()
    runs = api.runs("plant_disease_detection/" + project_name)
    animal_index = len(runs)
    return animal_names[animal_index % len(animal_names)]


class DiaMOSDataset(Dataset):
    def __init__(self, csv_file, img_dir, data_path, transform=None, imputation_value=-1):
        self.data = []
        self.img_dir = img_dir
        self.transform = transform
        self.imputation_value = imputation_value
        
        csv_file_path = os.path.join(data_path, 'annotation/csv', csv_file) 
        
        with open(csv_file_path, 'r') as f:
            lines = f.readlines()
            header = lines[0].strip().split(';')
            for line in lines[1:]:
                datapoint = dict(zip(header, line.strip().split(';')))
                
                disease = []
                disease.append(int(datapoint['healthy']))
                disease.append(int(datapoint['pear_slug']))
                disease.append(int(datapoint['leaf_spot']))
                disease.append(int(datapoint['curl']))

                severity = []
                for i in range(5):
                    value = datapoint[f'severity_{i}']
                    if value.lower() == 'not estimable':
                        severity.append(self.imputation_value)
                    else:
                        severity.append(int(value))

                self.data.append((datapoint['filename'], disease, severity))


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        filename, disease, severity = self.data[idx]

        image_path = None
        for subfolder in ['curl', 'healthy', 'slug', 'spot']:
            potential_path = os.path.join(self.img_dir, subfolder, filename)
            if os.path.exists(potential_path):
                image_path = potential_path
                break

        if image_path is None:
            raise Exception(f"Image not found: {filename}")

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        image = Image.fromarray(image) 
        if self.transform:
            image = self.transform(image)

        disease_label = torch.tensor(disease, dtype=torch.int)
        severity_label = torch.tensor(severity, dtype=torch.int)

        return image, disease_label, severity_label


class DiaMOSDataset_Cartesian(Dataset):
    def __init__(self, csv_file, img_dir, data_path, transform=None, imputation_value=-1):
        self.data = []
        self.img_dir = img_dir
        self.transform = transform
        self.imputation_value = imputation_value
        self.label_mapping = {}  
        self.label_counter = 0

        csv_file_path = os.path.join(data_path, 'annotation/csv', csv_file) 
        
        with open(csv_file_path, 'r') as f:
            lines = f.readlines()
            header = lines[0].strip().split(';')
            for line in lines[1:]:
                datapoint = dict(zip(header, line.strip().split(';')))

                disease = np.argmax([int(datapoint['healthy']), int(datapoint['pear_slug']), 
                                     int(datapoint['leaf_spot']), int(datapoint['curl'])])

                severity = []
                for i in range(5):
                    value = datapoint[f'severity_{i}']
                    if value.lower() == 'not estimable':
                        severity.append(self.imputation_value)
                    else:
                        severity.append(int(value))
                severity = np.argmax(severity)

                combined_label = (disease, severity)
                if combined_label not in self.label_mapping:
                    self.label_mapping[combined_label] = self.label_counter
                    self.label_counter += 1

                self.data.append((datapoint['filename'], self.label_mapping[combined_label]))

        # Create the reverse mapping
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, combined_label = self.data[idx]

        image_path = None
        for subfolder in ['curl', 'healthy', 'slug', 'spot']:
            potential_path = os.path.join(self.img_dir, subfolder, filename)
            if os.path.exists(potential_path):
                image_path = potential_path
                break

        if image_path is None:
            raise Exception(f"Image not found: {filename}")

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        image = Image.fromarray(image) 
        if self.transform:
            image = self.transform(image)

        return image, combined_label
    

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func


    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0


    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def sanity_check_datasets():
    # For DiaMOSDataset
    dataset = DiaMOSDataset(csv_file='diaMOSPlant.csv', img_dir='data\\Pear\\leaves\\', data_path='data\\Pear\\')
    print(len(dataset))
    
    for i in range(10):
        image, disease_label, severity_label = dataset[i]
        print(f"Disease label: {disease_label}, Severity label: {severity_label}")

    # For DiaMOSDataset_Cartesian
    dataset = DiaMOSDataset_Cartesian(csv_file='diaMOSPlant.csv', img_dir='data\\Pear\\leaves\\', data_path='data\\Pear\\')
    print(len(dataset))
    for i in range(10):
        image, combined_label = dataset[i]
        print(f"Combined label: {combined_label}")

    disease_labels = ["Healthy", "Pear Slug", "Leaf Spot", "Curl"]
    severity_labels = ["None", "Low", "Medium", "High", "Very High"]

    # For DiaMOSDataset_Cartesian
    dataset = DiaMOSDataset_Cartesian(csv_file='diaMOSPlant.csv', img_dir='data\\Pear\\leaves\\', data_path='data\\Pear\\')
    print(len(dataset))
    for i in range(10):
        image, combined_label = dataset[i]
        disease_label = disease_labels[combined_label // len(severity_labels)]
        severity_label = severity_labels[combined_label % len(severity_labels)]
        print(f"Disease label: {disease_label}, Severity label: {severity_label}")

    import collections

    # Get the labels
    labels = [label for _, label in dataset]

    # Count the labels
    label_counts = collections.Counter(labels)

    # Print the label counts
    for label, count in label_counts.items():
        print(f"Label {label}: {count} instances")
