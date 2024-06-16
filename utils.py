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
import random
import torchvision.transforms.functional as F
import math
from wandb import Api
import traceback

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

possible_augmentations = [
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.9),
    transforms.RandomRotation(degrees=20),
    transforms.RandomResizedCrop(size=(256, 256), scale=(0.9, 1.2)),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2)),
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.2), shear=15)
]

class ReflectPadding:
    def __init__(self, padding=50):
        self.padding = padding

    def __call__(self, img):
        return F.pad(img, (self.padding, self.padding, self.padding, self.padding), padding_mode='edge')

def random_augmentations(image, possible_augmentations, padding=10):
    n = int(np.random.normal(loc=len(possible_augmentations)/2, scale=len(possible_augmentations)/4))
    n = max(0, min(n, len(possible_augmentations)))
    selected_augmentations = random.sample(possible_augmentations, n)
    random.shuffle(selected_augmentations)
    transform = transforms.Compose(selected_augmentations)
    return ReflectPadding(padding=padding)(transform(image))

def test_augmentations(image, possible_augmentations, num_images=20, images_per_row=8):
    num_rows = math.ceil(num_images / images_per_row)
    fig, ax = plt.subplots(num_rows, images_per_row, figsize=(20, 4*num_rows))
    ax = ax.flatten()
    ax[0].imshow(image)
    ax[0].set_title('Original Image')

    for i in range(1, num_images):
        # augmented_image = random_augmentations(random_augmentations(image, possible_augmentations), possible_augmentations)
        augmented_image = random_augmentations(image, possible_augmentations)
        ax[i].imshow(augmented_image)
        ax[i].set_title(f'Augmented Image {i}')
        ax[i].set_xlabel('image')

    ax[num_images].axis('off')
    plt.tight_layout()
    plt.show()

# test_augmentations(Image.open(f'data\Pear\\fruits\\20.jpg'), possible_augmentations, num_images=20, images_per_row=8)

class DiaMOSDataset(Dataset):
    def __init__(self, csv_file, img_dir, data_path, 
                 transform=None, 
                 augment=False, 
                 imputation_value=-1,
                 target_size=10000,
                 aug_dir='/kaggle/working/augmented/'):
        self.data = []
        self.img_dir = img_dir
        self.transform = transform
        self.augment = augment
        self.imputation_value = imputation_value
        self.target_size = target_size
        self.aug_dir = aug_dir

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
            potential_path = os.path.join(self.aug_dir, filename)
            if os.path.exists(potential_path):
                image_path = potential_path
            else:
                raise Exception(f"Image not found: {filename}")

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        image = Image.fromarray(image) 

        if self.transform:
            image = self.transform(image)

        disease_label = torch.tensor(disease, dtype=torch.int)
        severity_label = torch.tensor(severity, dtype=torch.int)

        return image, disease_label, severity_label
    

    def augment_dataset(self, target_size, save_dir='/kaggle/working/augmented/'):
        augmented_data = self.data.copy()
        current_size = len(self.data)
        os.makedirs(save_dir, exist_ok=True)

        while current_size < target_size:
            idx = random.randint(0, len(self.data) - 1)
            filename, disease, severity = self.data[idx]

            image_path = None
            for subfolder in ['curl', 'healthy', 'slug', 'spot']:
                potential_path = os.path.join(self.img_dir, subfolder, filename)
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break
            
            if image_path is None:
                potential_path = os.path.join(self.aug_dir, filename)
                if os.path.exists(potential_path):
                    image_path = potential_path
                else:
                    continue  # Skip if the image is not found

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            augmented_image = random_augmentations(image, possible_augmentations)

            augmented_filename = f"augmented_{current_size}.jpg"
            augmented_filepath = os.path.join(self.aug_dir, augmented_filename)
            try:
                augmented_image_np = np.array(augmented_image)
                augmented_image_bgr = cv2.cvtColor(augmented_image_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(augmented_filepath, augmented_image_bgr)
            except TypeError as e:
                print(f"Error saving image: {e}")
                print(f"Image type: {type(augmented_image)}")
                print(traceback.format_exc())
                continue 
            augmented_data.append((augmented_filename, disease, severity))
            current_size += 1
        
        self.data = augmented_data
            


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
