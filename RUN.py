import sys
import wandb
from torchvision import transforms
from types import SimpleNamespace
import torch
from torch import nn
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, random_split
import os
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
from wandb import Api
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import random
import torchvision.transforms.functional as F
import math
import traceback
from efficientnet_pytorch import EfficientNet
from torchvision.models import densenet201
from torchvision.models import mobilenet_v2
from torchvision.models import inception_v3
from torchvision.models import resnet50



#### utils.py

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



#### --------------------------------------------
#### separate_classifiers.py

class DiseaseModel(nn.Module):
    def __init__(self, num_disease_classes):
        super(DiseaseModel, self).__init__()
        vgg19 = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True)
        for param in vgg19.features.parameters():
            param.requires_grad = False
        self.features = vgg19.features
        self.avgpool = vgg19.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_disease_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class DiseaseModelResNet(nn.Module):
    def __init__(self, num_disease_classes):
        super(DiseaseModelResNet, self).__init__()
        resnet = resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_disease_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class DiseaseModelDenseNet(nn.Module):
    def __init__(self, num_disease_classes):
        super(DiseaseModelDenseNet, self).__init__()
        densenet = densenet201(pretrained=True)
        for param in densenet.parameters():
            param.requires_grad = False
        self.features = densenet.features
        self.classifier = nn.Sequential(
            nn.Linear(1920 * 7 * 7, 4096),  # Update the input size of the first linear layer
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_disease_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Reshape the output to a 4D tensor
        x = self.classifier(x)
        return x

class DiseaseModelInception(nn.Module):
    def __init__(self, num_disease_classes):
        super(DiseaseModelInception, self).__init__()
        self.inception = inception_v3(pretrained=True, aux_logits=False)  # Disable aux_logits

        # Freeze the convolutional base
        for param in self.inception.parameters():
            param.requires_grad = False

        # Replace the final fully connected layer
        in_features_main = self.inception.fc.in_features
        self.inception.fc = nn.Sequential(
            nn.Linear(in_features_main, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_disease_classes)
        )

    def forward(self, x):
        x = self.inception(x)
        return x 
    

class DiseaseModelMobile(nn.Module):
    def __init__(self, num_disease_classes):
        super(DiseaseModelMobile, self).__init__()
        mobilenet = mobilenet_v2(pretrained=True)
        for param in mobilenet.parameters():
            param.requires_grad = False
        self.features = mobilenet.features
        self.classifier = nn.Sequential(
            nn.Linear(62720, 4096),  # Adjusted input size
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_disease_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Reshape the output to a 4D tensor
        x = self.classifier(x)
        return x

class SeverityModel(nn.Module):
    def __init__(self, num_severity_levels):
        super(SeverityModel, self).__init__()
        vgg19 = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True)
        for param in vgg19.features.parameters():
            param.requires_grad = False
        self.features = vgg19.features
        self.avgpool = vgg19.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_severity_levels)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SeverityModelResNet(nn.Module):
    def __init__(self, num_severity_levels):
        super(SeverityModelResNet, self).__init__()
        resnet = resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_severity_levels)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SeverityModelDenseNet(nn.Module):
    def __init__(self, num_severity_levels):
        super(SeverityModelDenseNet, self).__init__()
        densenet = densenet201(pretrained=True)
        for param in densenet.parameters():
            param.requires_grad = False
        self.features = densenet.features
        self.classifier = nn.Sequential(
            nn.Linear(1920 * 7 * 7, 4096),  # Correct input size
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_severity_levels)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SeverityModelInception(nn.Module):
    def __init__(self, num_severity_levels):
        super(SeverityModelInception, self).__init__()
        inception = inception_v3(pretrained=True, aux_logits=False)  # Disable aux_logits
        for param in inception.parameters():
            param.requires_grad = False
        self.features = nn.Sequential(*list(inception.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(inception.fc.in_features, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_severity_levels)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SeverityModelMobile(nn.Module):
    def __init__(self, num_severity_levels):
        super(SeverityModelMobile, self).__init__()
        mobilenet = mobilenet_v2(pretrained=True)
        for param in mobilenet.parameters():
            param.requires_grad = False
        self.features = mobilenet.features
        self.classifier = nn.Sequential(
            nn.Linear(1280, 4096),  # Correct input size
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_severity_levels)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



def run_experiment_separate(args):
    def train(args, train_data, val_data, test_data, device, run):
        disease_model = args.get("disease_model", DiseaseModel(num_disease_classes=4)).to(device)
        severity_model = args.get("severity_model", SeverityModel(num_severity_levels=5)).to(device)
        run.watch(disease_model)
        run.watch(severity_model)

        disease_criterion = args.get("criterion", nn.CrossEntropyLoss())
        disease_optimizer = args.get("optimizer", torch.optim.Adam(disease_model.parameters(), lr=args.get("lr", 0.001)))

        severity_criterion = args.get("criterion", nn.CrossEntropyLoss())
        severity_optimizer = args.get("optimizer", torch.optim.Adam(severity_model.parameters(), lr=args.get("lr", 0.001)))

        avg_val_loss = 0

        # add a scheduler
        disease_scheduler = StepLR(disease_optimizer, step_size=10, gamma=0.1)
        severity_scheduler = StepLR(severity_optimizer, step_size=10, gamma=0.1)

        epochs = args.get("epochs", 10)
        best_val_loss = float('inf')  # Initialize the best validation loss

        total_t0 = time.time()
        early_stopping = EarlyStopping(patience=4, verbose=True)

        for epoch in range(epochs):
            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Put the model into training mode. Don't be mislead--the call to
            # `train` just changes the *mode*, it doesn't *perform* the training.
            # `dropout` and `batchnorm` layers behave differently during training
            # vs. test (source: https://stackoverflow.com/questimport gensim.downloader as api
            disease_model.train()
            severity_model.train()

            # Reset the total loss for this epoch.
            total_train_accuracy_disease = 0
            total_train_accuracy_severity = 0
            running_loss = 0.0

            # for images, disease_labels, severity_labels in train_data:
            # For each batch of training data...
            for step, batch in enumerate(train_data):

                # Progress update every 4 batches
                if step % 4 == 0 and not step == 0:
                    # Calculate elapsed time in minutes
                    elapsed = format_time(time.time() - t0)
                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_data), elapsed))

                images = batch[0].to(device)
                disease_labels = batch[1].to(device)
                severity_labels = batch[2].to(device)
                # images, disease_labels, severity_labels = images.to(device), disease_labels.to(device), severity_labels.to(device)

                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because
                # accumulating the gradients is "convenient while training RNNs".
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                disease_optimizer.zero_grad()
                severity_optimizer.zero_grad()

                disease_output = disease_model(images)
                severity_output = severity_model(images)

                disease_loss = disease_criterion(disease_output, torch.argmax(disease_labels, dim=1))
                severity_loss = severity_criterion(severity_output, torch.argmax(severity_labels, dim=1))

                loss = disease_loss + severity_loss
                # run.log({"loss": loss.item(), "val_loss": avg_val_loss})
                loss.backward()
                disease_optimizer.step()
                severity_optimizer.step()
                running_loss += loss.item()

                logits_disease = disease_output.detach().cpu().numpy()
                logits_severity = severity_output.detach().cpu().numpy()

                disease_label_ids = disease_labels.to('cpu').numpy()
                severity_label_ids = severity_labels.to('cpu').numpy()

                total_train_accuracy_disease += flat_accuracy(logits_disease, disease_label_ids)
                total_train_accuracy_severity += flat_accuracy(logits_severity, severity_label_ids)
            
            print("Epoch {}/{} - Training: ".format(epoch + 1, epochs))
            avg_loss = running_loss / len(train_data)
            print("\tAverage training loss: {0:.2f}".format(avg_loss))
            avg_train_accuracy_disease = total_train_accuracy_disease / len(train_data)
            avg_train_accuracy_severity = total_train_accuracy_severity / len(train_data)
            print("\tTrain Accuracy Disease: {0:.2f}".format(avg_train_accuracy_disease))
            print("\tTrain Accuracy Severity: {0:.2f}".format(avg_train_accuracy_severity))

            run.log({
                "train_loss": avg_loss,
                "train_accuracy_disease": avg_train_accuracy_disease,
                "train_accuracy_severity": avg_train_accuracy_severity,
            })

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            disease_model.eval()
            severity_model.eval()

            # with torch.no_grad():
            val_loss = 0.0
            # for images, disease_labels, severity_labels in val_data:
            # For each batch of training data...

            total_val_accuracy_disease = 0
            total_val_accuracy_severity = 0

            for _, batch in enumerate(val_data):
                images = batch[0].to(device)
                disease_labels = batch[1].to(device)
                severity_labels = batch[2].to(device)

                # images, disease_labels, severity_labels = images.to(device), disease_labels.to(device), severity_labels.to(device)
                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():
                    disease_output = disease_model(images)
                    severity_output = severity_model(images)

                disease_loss = disease_criterion(disease_output, torch.argmax(disease_labels, dim=1))
                severity_loss = severity_criterion(severity_output, torch.argmax(severity_labels, dim=1))
                val_loss += (disease_loss + severity_loss).item()

                logits_disease = disease_output.detach().cpu().numpy()
                logits_severity = severity_output.detach().cpu().numpy()

                disease_label_ids = disease_labels.to('cpu').numpy()
                severity_label_ids = severity_labels.to('cpu').numpy()

                total_val_accuracy_disease += flat_accuracy(logits_disease, disease_label_ids)
                total_val_accuracy_severity += flat_accuracy(logits_severity, severity_label_ids)

            print("Epoch {}/{} - Validation: ".format(epoch + 1, epochs))

            avg_val_loss = val_loss / len(val_data)
            print("\tAverage validation loss: {0:.2f}".format(avg_val_loss))

            avg_val_accuracy_disease = total_val_accuracy_disease / len(val_data)
            avg_val_accuracy_severity = total_val_accuracy_severity / len(val_data)
            print("\tValidation Accuracy Disease: {0:.2f}".format(avg_val_accuracy_disease))
            print("\tValidation Accuracy Severity: {0:.2f}".format(avg_val_accuracy_severity))

            # Log metrics to wandb
            run.log({
                "val_loss": avg_val_loss,
                "val_accuracy_disease": avg_val_accuracy_disease,
                "val_accuracy_severity": avg_val_accuracy_severity,
            })

            # Check if validation loss has improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(disease_model.state_dict(), "disease_model.pth")  # Save the model
                torch.save(severity_model.state_dict(), "severity_model.pth")  # Save the model
                run.save("disease_model.pth")
                run.save("severity_model.pth")

            # Perform early stopping
            early_stopping(avg_val_loss, disease_model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # Measure how long the validation run took.
            validation_time = format_time(time.time() - t0)

            print(" Validation Loss: {0:.2f}".format(avg_val_loss))
            print(" Validation took: {:}".format(validation_time))

            disease_scheduler.step()
            severity_scheduler.step()

        # Load the best model
        disease_model.load_state_dict(torch.load("disease_model.pth"))
        severity_model.load_state_dict(torch.load("severity_model.pth"))

        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

        return disease_model, severity_model

    # ------------------------------
    # ----- Data Preparation -------
    # ------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transformations = args.transforms if hasattr(args, 'transforms') else transforms.Compose([])

    name = args.name if hasattr(args, 'name') else "separate_classifiers"

    full_dataset = DiaMOSDataset(csv_file=args.csv_file if hasattr(args, 'csv_file') else 'diaMOSPlant.csv',
                                 img_dir=args.img_dir if hasattr(args, 'img_dir') else '/kaggle/input/diamos-plant-dataset/Pear/leaves',
                                 data_path=args.data_path if hasattr(args, 'data_path') else '/kaggle/input/diamos-plant-dataset/Pear/',
                                 transform=transformations)

    train_size = args.train_size if hasattr(args, 'train_size') else int(0.6 * len(full_dataset))
    val_size = args.val_size if hasattr(args, 'val_size') else int(0.2 * len(full_dataset))
    test_size = args.test_size if hasattr(args, 'test_size') else len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    augmentation = args.augmentation if hasattr(args, 'augmentation') else False
    # augment_operations = args.augment_operations if hasattr(args, 'augment_operations') else []
    augment_target_size_factor = args.augment_target_size_factor if hasattr(args, 'augment_target_size_factor') else 1
    augment_save_dir = args.augment_save_dir if hasattr(args, 'augment_save_dir') else '/kaggle/working/augmented/'

    if augmentation:
        print("Augmenting dataset...")
        print("Length of train dataset before augmentation: ", len(train_dataset))
        train_dataset.dataset.augment = True
        train_dataset.dataset.save_dir = augment_save_dir
        train_dataset.dataset.augment_dataset(len(train_dataset) * augment_target_size_factor, augment_save_dir)
        print("Length of train dataset after augmentation: ", len(train_dataset))

    batch_size = args.batch_size if hasattr(args, 'batch_size') else 16
    num_workers = args.num_workers if hasattr(args, 'num_workers') else 2

    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # ------------------------------
    # ----- Model Training ---------
    # ------------------------------

    args_dict = vars(args)
    project_name = args.project_name if hasattr(args, 'project_name') else "plant_disease_detection"
    run_name = name + "_" + animal_version_name(project_name)
    print("Project Name:", project_name)
    run = wandb.init(name=run_name, reinit=True, entity="plant_disease_detection", project=project_name)
    serializable_args_dict = {k: v for k, v in args_dict.items() if isinstance(v, (int, float, str, bool, list, dict, tuple, set))}
    run.config.update(serializable_args_dict)
    disease_model, severity_model = train(args_dict, train_data, val_data, test_data, device, run)

    if augmentation:
        print("Removing augmented images...")
        os.rmdir(augment_save_dir)
        print("Augmented images removed!")

    # ------------------------------
    # ----- Model Evaluation -------
    # ------------------------------


    # Evaluation loop (for validation or test data)
    disease_model.eval()
    severity_model.eval()

    # Add code for evaluation here (using validation or test data)
    total_disease_correct = 0
    total_severity_correct = 0
    total_samples = 0

    # Initialize lists to store correctly classified images (to print them in the next cell)
    correct_images = []
    correct_disease_labels = []
    correct_severity_labels = []
    correct_disease_predictions = []
    correct_severity_predictions = []

    all_disease_labels = []
    all_disease_predictions = []
    all_severity_labels = []
    all_severity_predictions = []

    with torch.no_grad():
        for images, disease_labels, severity_labels in test_data:
            images, disease_labels, severity_labels = images.to(device), disease_labels.to(device), severity_labels.to(device)
            disease_output = disease_model(images)
            severity_output = severity_model(images)

            # Disease classification
            _, disease_predicted = torch.max(disease_output, 1)
            disease_correct = (disease_predicted == torch.argmax(disease_labels, dim=1))
            total_disease_correct += disease_correct.sum().item()

            # Severity classification
            _, severity_predicted = torch.max(severity_output, 1)
            severity_correct = (severity_predicted == torch.argmax(severity_labels, dim=1))
            total_severity_correct += severity_correct.sum().item()

            total_samples += images.size(0)

            # Find indices of correctly classified images
            correct_indices = (disease_correct & severity_correct).nonzero().squeeze()

            # Check if correct_indices is a scalar
            if correct_indices.dim() == 0:
                correct_indices = [correct_indices.item()]

            # Append correctly classified images and their labels/predictions
            correct_images.extend(images[correct_indices].cpu().numpy())
            correct_disease_labels.extend(disease_labels[correct_indices].cpu().numpy())
            correct_severity_labels.extend(severity_labels[correct_indices].cpu().numpy())
            correct_disease_predictions.extend(disease_predicted[correct_indices].cpu().numpy())
            correct_severity_predictions.extend(severity_predicted[correct_indices].cpu().numpy())

            # Append all disease labels and predictions
            all_disease_labels.extend(torch.argmax(disease_labels, dim=1).cpu().numpy())
            all_disease_predictions.extend(disease_predicted.cpu().numpy())
            all_severity_labels.extend(torch.argmax(severity_labels, dim=1).cpu().numpy())
            all_severity_predictions.extend(severity_predicted.cpu().numpy())

    disease_accuracy = total_disease_correct / total_samples
    severity_accuracy = total_severity_correct / total_samples
    print(f'Disease Classification Accuracy: {disease_accuracy * 100:.2f}%')
    print(f'Severity Classification Accuracy: {severity_accuracy * 100:.2f}%')

    run.log({
        "test_accuracy_disease": disease_accuracy,
        "test_accuracy_severity": severity_accuracy,
    })

    disease_precision = precision_score(all_disease_labels, all_disease_predictions, average='weighted', zero_division=1)
    severity_precision = precision_score(all_severity_labels, all_severity_predictions, average='weighted', zero_division=1)
    disease_recall = recall_score(all_disease_labels, all_disease_predictions, average='weighted', zero_division=1)
    severity_recall = recall_score(all_severity_labels, all_severity_predictions, average='weighted', zero_division=1)
    disease_f1 = f1_score(all_disease_labels, all_disease_predictions, average='weighted', zero_division=1)
    severity_f1 = f1_score(all_severity_labels, all_severity_predictions, average='weighted', zero_division=1)

    # Log metrics to wandb
    run.log({
        "test_precision_disease": disease_precision,
        "test_precision_severity": severity_precision,
        "test_recall_disease": disease_recall,
        "test_recall_severity": severity_recall,
        "test_f1_disease": disease_f1,
        "test_f1_severity": severity_f1,
    })

    info = {
        "run_name": run_name,
        "accuracy_disease": disease_accuracy,
        "accuracy_severity": severity_accuracy,
        "precision_disease": disease_precision,
        "precision_severity": severity_precision,
        "recall_disease": disease_recall,
        "recall_severity": severity_recall,
        "f1_disease": disease_f1,
        "f1_severity": severity_f1,
    }

    print("Metrics succesfuly logged to wandb!")



    # Save the trained model
    torch.save(disease_model.state_dict(), "disease_model.pth")
    torch.save(severity_model.state_dict(), "severity_model.pth")
    run.save("disease_model.pth")
    run.save("severity_model.pth")

    with open("model_info.txt", "w") as f:
        f.write(json.dumps(info, indent=4))

    wandb.join()



#### --------------------------------------------
#### combined_classifier.py

def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]


class DiseaseSeverityModel_CombinedVGG(nn.Module):
    def __init__(self, num_combined_labels):
        super(DiseaseSeverityModel_CombinedVGG, self).__init__()
        vgg19 = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True)
        for param in vgg19.features.parameters(): param.requires_grad = False
        self.features = vgg19.features
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_combined_labels)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    
class DiseaseSeverityModel_CombinedResNet50(nn.Module):
    def __init__(self, num_combined_labels):
        super(DiseaseSeverityModel_CombinedResNet50, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        for param in resnet50.parameters(): param.requires_grad = False
        self.features = nn.Sequential(*list(resnet50.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_combined_labels)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class DiseaseSeverityModel_CombinedV1(nn.Module):
    def __init__(self, num_combined_labels):
        super(DiseaseSeverityModel_CombinedV1, self).__init__()
        vgg19 = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True)
        for param in vgg19.features.parameters(): param.requires_grad = False
        self.features = vgg19.features
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_combined_labels)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def run_experiment_combined(args):    
    def train(args, train_data, val_data, test_data, device, run, reverse_label_mapping):    
        num_combined_labels = 20  # Adjust this according to your data
        model = args.get("model", DiseaseSeverityModel_CombinedV1(num_combined_labels))
        model.to(device)
        run.watch(model)

        criterion = args.get("criterion", nn.CrossEntropyLoss())  
        optimizer = args.get("optimizer", torch.optim.Adam(model.parameters(), lr=args.get("lr", 0.001)))
        avg_val_loss = 0

        # add a scheduler
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

        epochs = args.get("epochs", 10)
        best_val_loss = float('inf')  # Initialize the best validation loss

        early_stopping = EarlyStopping(patience=4, verbose=True)

        # Training loop
        for epoch in range(epochs):
            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
            print('Training...')
            
            # Measure how long the training epoch takes.
            t0 = time.time()
            
            # Put the model into training mode. Don't be mislead--the call to
            # `train` just changes the *mode*, it doesn't *perform* the training.
            # `dropout` and `batchnorm` layers behave differently during training
            # vs. test (source: https://stackoverflow.com/questimport gensim.downloader as api
            model.train()

            # Reset the total loss for this epoch.
            total_train_accuracy_disease = 0
            total_train_accuracy_severity = 0
            running_loss = 0.0

            for step, batch in enumerate(train_data):
                # Progress update every 4 batches
                if step % 4 == 0 and not step == 0:
                    # Calculate elapsed time in minutes
                    elapsed = format_time(time.time() - t0)
                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_data), elapsed))
                
                images = batch[0].to(device)
                combined_labels = batch[1].to(device)  # Assuming combined labels are at index 1
                
                optimizer.zero_grad()

                combined_output = model(images)
                
                loss = criterion(combined_output, combined_labels)
                
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                logits_combined = combined_output.detach().cpu().numpy()
                combined_label_ids = combined_labels.to('cpu').numpy()
                
                # Get disease and severity labels
                labels_disease, labels_severity = zip(*[reverse_label_mapping[label] for label in combined_label_ids])
                labels_disease = torch.tensor(labels_disease).to(device)
                labels_severity = torch.tensor(labels_severity).to(device)

                # Convert labels to one-hot encoding
                labels_disease_one_hot = one_hot_encode(labels_disease.cpu().numpy(), num_combined_labels)
                labels_severity_one_hot = one_hot_encode(labels_severity.cpu().numpy(), num_combined_labels)

                # Split the combined logits into disease and severity parts
                half_dim = logits_combined.shape[1] // 2
                logits_disease, logits_severity = logits_combined[:, :half_dim], logits_combined[:, half_dim:]

                total_train_accuracy_disease += flat_accuracy(logits_disease, labels_disease_one_hot)
                total_train_accuracy_severity += flat_accuracy(logits_severity, labels_severity_one_hot)

            print("Epoch {}/{} - Training: ".format(epoch + 1, epochs))

            avg_loss = running_loss / len(train_data)
            print("\tAverage training loss: {0:.2f}".format(avg_loss))
            avg_train_accuracy_disease = total_train_accuracy_disease / len(train_data)
            avg_train_accuracy_severity = total_train_accuracy_severity / len(train_data)
            print("\tTrain Accuracy Disease: {0:.2f}".format(avg_train_accuracy_disease))
            print("\tTrain Accuracy Severity: {0:.2f}".format(avg_train_accuracy_severity))

            run.log({
                "train_loss": avg_loss,
                "train_accuracy_disease": avg_train_accuracy_disease,
                "train_accuracy_severity": avg_train_accuracy_severity,
            })

            print("")
            print("Running Validation...")

            t0 = time.time()
            model.eval()

            total_val_accuracy_disease = 0
            total_val_accuracy_severity = 0
            total_val_loss = 0

            for _, batch in enumerate(val_data):   
                    
                images = batch[0].to(device)
                combined_labels = batch[1].to(device)  # Assuming combined labels are at index 1
                
                with torch.no_grad():
                    combined_output = model(images)
                    
                loss = criterion(combined_output, combined_labels)
                total_val_loss += loss.item()

                logits_combined = combined_output.detach().cpu().numpy()
                combined_label_ids = combined_labels.to('cpu').numpy()
                
                # Get disease and severity labels
                labels_disease, labels_severity = zip(*[reverse_label_mapping[label] for label in combined_label_ids])
                labels_disease = torch.tensor(labels_disease).to(device)
                labels_severity = torch.tensor(labels_severity).to(device)
                
                # Convert labels to one-hot encoding
                labels_disease_one_hot = one_hot_encode(labels_disease.cpu().numpy(), num_combined_labels)
                labels_severity_one_hot = one_hot_encode(labels_severity.cpu().numpy(), num_combined_labels)

                # Split the combined logits into disease and severity parts
                half_dim = logits_combined.shape[1] // 2
                logits_disease, logits_severity = logits_combined[:, :half_dim], logits_combined[:, half_dim:]

                total_val_accuracy_disease += flat_accuracy(logits_disease, labels_disease_one_hot)
                total_val_accuracy_severity += flat_accuracy(logits_severity, labels_severity_one_hot)
            
            print("Epoch {}/{} - Validation: ".format(epoch + 1, epochs))

            avg_val_loss = total_val_loss / len(val_data)
            print("\tAverage validation loss: {0:.2f}".format(avg_val_loss))

            avg_val_accuracy_disease = total_val_accuracy_disease / len(val_data)
            avg_val_accuracy_severity = total_val_accuracy_severity / len(val_data)
            print("\tValidation Accuracy Disease: {0:.2f}".format(avg_val_accuracy_disease))
            print("\tValidation Accuracy Severity: {0:.2f}".format(avg_val_accuracy_severity))

            # Log metrics to wandb
            run.log({
                "val_loss": avg_val_loss,
                "val_accuracy_disease": avg_val_accuracy_disease,
                "val_accuracy_severity": avg_val_accuracy_severity,
            })

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), 'best_model.pt')

            scheduler.step()

            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print("")
        print("Training complete!")

        # Load the best state dictionary into a new model
        best_model = DiseaseSeverityModel_CombinedV1(num_combined_labels)
        best_model.load_state_dict(torch.load('best_model.pt'))
        best_model.to(device)

        return best_model


    # ------------------------------
    # ----- Data Preparation -------
    # ------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transformations = args.transforms if hasattr(args, 'transforms') else transforms.Compose([])
    
    name = args.name if hasattr(args, 'name') else "combined_classifier"

    full_dataset = DiaMOSDataset_Cartesian(csv_file=args.csv_file if hasattr(args, 'csv_file') else 'diaMOSPlant.csv', 
                                 img_dir=args.img_dir if hasattr(args, 'img_dir') else '/kaggle/input/diamos-plant-dataset/Pear/leaves', 
                                 data_path=args.data_path if hasattr(args, 'data_path') else '/kaggle/input/diamos-plant-dataset/Pear/', 
                                 transform=transformations)
    
    train_size = args.train_size if hasattr(args, 'train_size') else int(0.6 * len(full_dataset))
    val_size = args.val_size if hasattr(args, 'val_size') else int(0.2 * len(full_dataset))
    test_size = args.test_size if hasattr(args, 'test_size') else len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    batch_size = args.batch_size if hasattr(args, 'batch_size') else 16
    num_workers = args.num_workers if hasattr(args, 'num_workers') else 2

    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # ------------------------------
    # ----- Model Training ---------
    # ------------------------------

    args_dict = vars(args)
    project_name = args.project_name if hasattr(args, 'project_name') else "plant_disease_detection"
    run_name = name + "_" + animal_version_name(project_name)
    print("Project Name:", project_name)
    run = wandb.init(name=run_name, reinit=True, entity="plant_disease_detection", project=project_name)
    serializable_args_dict = {k: v for k, v in args_dict.items() if isinstance(v, (int, float, str, bool, list, dict, tuple, set))}
    run.config.update(serializable_args_dict)
    model = train(args_dict, train_data, val_data, test_data, device, run, full_dataset.reverse_label_mapping)

    # ------------------------------
    # ----- Model Evaluation -------
    # ------------------------------

    # Evaluation loop (for validation or test data)
    model.eval()

    # Add code for evaluation here (using validation or test data)
    total_disease_correct = 0
    total_severity_correct = 0
    total_samples = 0

    # Initialize lists to store correctly classified images (to print them in the next cell)
    correct_images = []
    correct_disease_labels = []
    correct_severity_labels = []
    correct_disease_predictions = []
    correct_severity_predictions = []

    all_disease_labels = []
    all_disease_predictions = []
    all_severity_labels = []
    all_severity_predictions = []

    with torch.no_grad():
        for images, labels in test_data:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            
            # Split labels into disease and severity
            disease_labels = labels // 5  # Assuming there are 4 severity levels
            severity_labels = labels % 5  # Assuming there are 4 severity levels

            # Classification
            _, predicted = torch.max(output, 1)
            disease_predicted = predicted // 5  # Assuming there are 4 severity levels
            severity_predicted = predicted % 5  # Assuming there are 4 severity levels

            disease_correct = (disease_predicted == disease_labels)
            total_disease_correct += disease_correct.sum().item()

            severity_correct = (severity_predicted == severity_labels)
            total_severity_correct += severity_correct.sum().item()
            
            total_samples += images.size(0)

            # Find indices of correctly classified images
            correct_indices = (disease_correct & severity_correct).nonzero().squeeze()

            # Check if correct_indices is a scalar
            if correct_indices.dim() == 0:
                correct_indices = [correct_indices.item()]

            # Append correctly classified images and their labels/predictions
            correct_images.extend(images[correct_indices].cpu().numpy())
            correct_disease_labels.extend(disease_labels[correct_indices].cpu().numpy())
            correct_severity_labels.extend(severity_labels[correct_indices].cpu().numpy())
            correct_disease_predictions.extend(disease_predicted[correct_indices].cpu().numpy())
            correct_severity_predictions.extend(severity_predicted[correct_indices].cpu().numpy())

            # Append all disease labels and predictions
            all_disease_labels.extend(disease_labels.cpu().numpy())
            all_disease_predictions.extend(disease_predicted.cpu().numpy())
            all_severity_labels.extend(severity_labels.cpu().numpy())
            all_severity_predictions.extend(severity_predicted.cpu().numpy())

    disease_accuracy = total_disease_correct / total_samples
    severity_accuracy = total_severity_correct / total_samples
    print(f'Disease Classification Accuracy: {disease_accuracy * 100:.2f}%')
    print(f'Severity Classification Accuracy: {severity_accuracy * 100:.2f}%')

    run.log({
        "test_accuracy_disease": disease_accuracy,
        "test_accuracy_severity": severity_accuracy,
    })

    disease_precision = precision_score(all_disease_labels, all_disease_predictions, average='weighted', zero_division=1)
    severity_precision = precision_score(all_severity_labels, all_severity_predictions, average='weighted', zero_division=1)
    disease_recall = recall_score(all_disease_labels, all_disease_predictions, average='weighted', zero_division=1)
    severity_recall = recall_score(all_severity_labels, all_severity_predictions, average='weighted', zero_division=1)
    disease_f1 = f1_score(all_disease_labels, all_disease_predictions, average='weighted', zero_division=1)
    severity_f1 = f1_score(all_severity_labels, all_severity_predictions, average='weighted', zero_division=1)


    # Log metrics to wandb
    run.log({
        "test_precision_disease": disease_precision,
        "test_precision_severity": severity_precision,
        "test_recall_disease": disease_recall,
        "test_recall_severity": severity_recall,
        "test_f1_disease": disease_f1,
        "test_f1_severity": severity_f1,
    })

    info = {
        "run_name": run_name,
        "accuracy_disease": disease_accuracy,
        "accuracy_severity": severity_accuracy,
        "precision_disease": disease_precision,
        "precision_severity": severity_precision,
        "recall_disease": disease_recall,
        "recall_severity": severity_recall,
        "f1_disease": disease_f1,
        "f1_severity": severity_f1,
    }

    print("Metrics succesfuly logged to wandb!")



    # Save the trained model
    torch.save(model.state_dict(), "disease_severity_model.pth")
    run.save("disease_severity_model.pth")

    with open("model_info.txt", "w") as f:
        f.write(json.dumps(info, indent=4))

    wandb.join()



#### --------------------------------------------
#### divergent_heads_classifier.py


class DiseaseSeverityModelVGG(nn.Module):
    def __init__(self, num_disease_classes, num_severity_levels):
        super(DiseaseSeverityModelVGG, self).__init__()
        vgg19 = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True)
        for param in vgg19.features.parameters(): param.requires_grad = False
        self.features = vgg19.features
        self.avgpool = vgg19.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )
        self.disease_classifier = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_disease_classes),
        )
        self.severity_classifier = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_severity_levels),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        disease_output = self.disease_classifier(x)
        severity_output = self.severity_classifier(x)
        return disease_output, severity_output
    

class DiseaseSeverityModelResNet50(nn.Module):
    def __init__(self, num_disease_classes, num_severity_levels):
        super(DiseaseSeverityModelResNet50, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        for param in resnet50.parameters(): param.requires_grad = False
        self.features = nn.Sequential(*list(resnet50.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )
        self.disease_classifier = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_disease_classes),
        )
        self.severity_classifier = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_severity_levels),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        disease_output = self.disease_classifier(x)
        severity_output = self.severity_classifier(x)
        return disease_output, severity_output


class DiseaseSeverityModelV1(nn.Module):
    def __init__(self, num_disease_classes, num_severity_levels):
        super(DiseaseSeverityModelV1, self).__init__()
        vgg19 = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True)
        for param in vgg19.features.parameters(): param.requires_grad = False
        self.features = vgg19.features
        self.avgpool = vgg19.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )
        self.disease_classifier = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_disease_classes),
        )
        self.severity_classifier = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_severity_levels),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        disease_output = self.disease_classifier(x)
        severity_output = self.severity_classifier(x)
        return disease_output, severity_output


def run_experiment_divergent(args):
    def train(args, train_data, val_data, test_data, device, run):    
        model = args.get("model", DiseaseSeverityModelV1(num_disease_classes=4, num_severity_levels=5))
        model.to(device)
        run.watch(model)

        criterion = args.get("criterion", nn.CrossEntropyLoss())  
        optimizer = args.get("optimizer", torch.optim.Adam(model.parameters(), lr=args.get("lr", 0.001)))
        avg_val_loss = 0

        # add a scheduler
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

        epochs = args.get("epochs", 10)
        best_val_loss = float('inf')  # Initialize the best validation loss

        total_t0 = time.time()
        early_stopping = EarlyStopping(patience=4, verbose=True)

        for epoch in range(epochs):
            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
            print('Training...')
            
            # Measure how long the training epoch takes.
            t0 = time.time()
            
            # Put the model into training mode. Don't be mislead--the call to
            # `train` just changes the *mode*, it doesn't *perform* the training.
            # `dropout` and `batchnorm` layers behave differently during training
            # vs. test (source: https://stackoverflow.com/questimport gensim.downloader as api
            model.train()
            
            # Reset the total loss for this epoch.
            total_train_accuracy_disease = 0
            total_train_accuracy_severity = 0
            running_loss = 0.0
            
            # for images, disease_labels, severity_labels in train_data:
            # For each batch of training data...
            for step, batch in enumerate(train_data):
                
                # Progress update every 4 batches
                if step % 4 == 0 and not step == 0:
                    # Calculate elapsed time in minutes
                    elapsed = format_time(time.time() - t0)
                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_data), elapsed))
                
                images = batch[0].to(device)
                disease_labels = batch[1].to(device)
                severity_labels = batch[2].to(device)
                # images, disease_labels, severity_labels = images.to(device), disease_labels.to(device), severity_labels.to(device)
                
                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because
                # accumulating the gradients is "convenient while training RNNs".
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                optimizer.zero_grad()
                
                disease_output, severity_output = model(images)
                
                disease_loss = criterion(disease_output, torch.argmax(disease_labels, dim=1))
                severity_loss = criterion(severity_output, torch.argmax(severity_labels, dim=1))
                
                loss = disease_loss + severity_loss
                # run.log({"loss": loss.item(), "val_loss": avg_val_loss})
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
                
                logits_disease = disease_output.detach().cpu().numpy()
                logits_severity = severity_output.detach().cpu().numpy()
                
                disease_label_ids = disease_labels.to('cpu').numpy()
                severity_label_ids = severity_labels.to('cpu').numpy()
                
                total_train_accuracy_disease += flat_accuracy(logits_disease, disease_label_ids)
                total_train_accuracy_severity += flat_accuracy(logits_severity, severity_label_ids)

            print("Epoch {}/{} - Training: ".format(epoch + 1, epochs))
            avg_loss = running_loss / len(train_data)
            print("\tAverage training loss: {0:.2f}".format(avg_loss))
            avg_train_accuracy_disease = total_train_accuracy_disease / len(train_data)
            avg_train_accuracy_severity = total_train_accuracy_severity / len(train_data)
            print("\tTrain Accuracy Disease: {0:.2f}".format(avg_train_accuracy_disease))
            print("\tTrain Accuracy Severity: {0:.2f}".format(avg_train_accuracy_severity))
    
            run.log({
                "train_loss": avg_loss,
                "train_accuracy_disease": avg_train_accuracy_disease,
                "train_accuracy_severity": avg_train_accuracy_severity,
            })

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")
            
            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            model.eval()
            
            # with torch.no_grad():
            total_val_loss = 0.0
            # for images, disease_labels, severity_labels in val_data:
            # For each batch of training data...

            total_val_accuracy_disease = 0
            total_val_accuracy_severity = 0

            for _, batch in enumerate(val_data):   
                    
                images = batch[0].to(device)
                disease_labels = batch[1].to(device)
                severity_labels = batch[2].to(device)
                
                # images, disease_labels, severity_labels = images.to(device), disease_labels.to(device), severity_labels.to(device)
                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():
                    disease_output, severity_output = model(images)
                    
                disease_loss = criterion(disease_output, torch.argmax(disease_labels, dim=1))
                severity_loss = criterion(severity_output, torch.argmax(severity_labels, dim=1))
                total_val_loss += (disease_loss + severity_loss).item()

                logits_disease = disease_output.detach().cpu().numpy()
                logits_severity = severity_output.detach().cpu().numpy()
                
                disease_label_ids = disease_labels.to('cpu').numpy()
                severity_label_ids = severity_labels.to('cpu').numpy()
                
                total_val_accuracy_disease += flat_accuracy(logits_disease, disease_label_ids)
                total_val_accuracy_severity += flat_accuracy(logits_severity, severity_label_ids)

            print("Epoch {}/{} - Validation: ".format(epoch + 1, epochs))

            avg_val_loss = total_val_loss / len(val_data)
            print("\tAverage validation loss: {0:.2f}".format(avg_val_loss))

            avg_val_accuracy_disease = total_val_accuracy_disease / len(val_data)
            avg_val_accuracy_severity = total_val_accuracy_severity / len(val_data)
            print("\tValidation Accuracy Disease: {0:.2f}".format(avg_val_accuracy_disease))
            print("\tValidation Accuracy Severity: {0:.2f}".format(avg_val_accuracy_severity))
            
            # Log metrics to wandb
            run.log({
                "val_loss": avg_val_loss,
                "val_accuracy_disease": avg_val_accuracy_disease,
                "val_accuracy_severity": avg_val_accuracy_severity,
            })
            
            # Check if validation loss has improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "disease_severity_model.pth")  # Save the model
                run.save("disease_severity_model.pth")

            # Perform early stopping
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            # Measure how long the validation run took.
            validation_time = format_time(time.time() - t0)
            
            # print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("\tValidation took: {:}".format(validation_time))

            scheduler.step()


        # Load the best model
        model.load_state_dict(torch.load("disease_severity_model.pth"))

        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

        return model
    
    
    # ------------------------------
    # ----- Data Preparation -------
    # ------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    transformations = args.transforms if hasattr(args, 'transforms') else transforms.Compose([])
    
    name = args.name if hasattr(args, 'name') else "divergent_heads_classifier"

    full_dataset = DiaMOSDataset(csv_file=args.csv_file if hasattr(args, 'csv_file') else 'diaMOSPlant.csv', 
                                 img_dir=args.img_dir if hasattr(args, 'img_dir') else '/kaggle/input/diamos-plant-dataset/Pear/leaves', 
                                 data_path=args.data_path if hasattr(args, 'data_path') else '/kaggle/input/diamos-plant-dataset/Pear/', 
                                 transform=transformations)

    train_size = args.train_size if hasattr(args, 'train_size') else int(0.6 * len(full_dataset))
    val_size = args.val_size if hasattr(args, 'val_size') else int(0.2 * len(full_dataset))
    test_size = args.test_size if hasattr(args, 'test_size') else len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
        
    augmentation = args.augmentation if hasattr(args, 'augmentation') else False
    # augment_operations = args.augment_operations if hasattr(args, 'augment_operations') else []
    augment_target_size_factor = args.augment_target_size_factor if hasattr(args, 'augment_target_size_factor') else 1
    augment_save_dir = args.augment_save_dir if hasattr(args, 'augment_save_dir') else '/kaggle/working/augmented/'

    if augmentation:
        print("Augmenting dataset...")
        print("Length of train dataset before augmentation: ", len(train_dataset))
        train_dataset.dataset.augment = True
        train_dataset.dataset.save_dir = augment_save_dir
        train_dataset.dataset.augment_dataset(len(train_dataset) * augment_target_size_factor, augment_save_dir)
        print("Length of train dataset after augmentation: ", len(train_dataset))

    
    batch_size = args.batch_size if hasattr(args, 'batch_size') else 16
    num_workers = args.num_workers if hasattr(args, 'num_workers') else 2

    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # ------------------------------
    # ----- Model Training ---------
    # ------------------------------

    args_dict = vars(args)
    project_name = args.project_name if hasattr(args, 'project_name') else "plant_disease_detection"
    run_name = name + "_" + animal_version_name(project_name)
    print("Project Name:", project_name)
    run = wandb.init(name=run_name, reinit=True, entity="plant_disease_detection", project=project_name)
    serializable_args_dict = {k: v for k, v in args_dict.items() if isinstance(v, (int, float, str, bool, list, dict, tuple, set))}
    run.config.update(serializable_args_dict)
    model = train(args_dict, train_data, val_data, test_data, device, run)

    if augmentation:
        print("Removing augmented images...")
        os.rmdir(augment_save_dir)
        print("Augmented images removed!")

    # ------------------------------
    # ----- Model Evaluation -------
    # ------------------------------

    # Evaluation loop (for validation or test data)
    model.eval()

    # Add code for evaluation here (using validation or test data)
    total_disease_correct = 0
    total_severity_correct = 0
    total_samples = 0

    # Initialize lists to store correctly classified images (to print them in the next cell)
    correct_images = []
    correct_disease_labels = []
    correct_severity_labels = []
    correct_disease_predictions = []
    correct_severity_predictions = []

    all_disease_labels = []
    all_disease_predictions = []
    all_severity_labels = []
    all_severity_predictions = []

    with torch.no_grad():
        for images, disease_labels, severity_labels in test_data:
            images, disease_labels, severity_labels = images.to(device), disease_labels.to(device), severity_labels.to(device)
            disease_output, severity_output = model(images)
            
            # Disease classification
            _, disease_predicted = torch.max(disease_output, 1)
            disease_correct = (disease_predicted == torch.argmax(disease_labels, dim=1))
            total_disease_correct += disease_correct.sum().item()
            
            # Severity classification
            _, severity_predicted = torch.max(severity_output, 1)
            severity_correct = (severity_predicted == torch.argmax(severity_labels, dim=1))
            total_severity_correct += severity_correct.sum().item()
            
            total_samples += images.size(0)

            # Find indices of correctly classified images
            correct_indices = (disease_correct & severity_correct).nonzero().squeeze()

            # Check if correct_indices is a scalar
            if correct_indices.dim() == 0:
                correct_indices = [correct_indices.item()]

            # Append correctly classified images and their labels/predictions
            correct_images.extend(images[correct_indices].cpu().numpy())
            correct_disease_labels.extend(disease_labels[correct_indices].cpu().numpy())
            correct_severity_labels.extend(severity_labels[correct_indices].cpu().numpy())
            correct_disease_predictions.extend(disease_predicted[correct_indices].cpu().numpy())
            correct_severity_predictions.extend(severity_predicted[correct_indices].cpu().numpy())

            # Append all disease labels and predictions
            all_disease_labels.extend(torch.argmax(disease_labels, dim=1).cpu().numpy())
            all_disease_predictions.extend(disease_predicted.cpu().numpy())
            all_severity_labels.extend(torch.argmax(severity_labels, dim=1).cpu().numpy())
            all_severity_predictions.extend(severity_predicted.cpu().numpy())

    disease_accuracy = total_disease_correct / total_samples
    severity_accuracy = total_severity_correct / total_samples
    print(f'Disease Classification Accuracy: {disease_accuracy * 100:.2f}%')
    print(f'Severity Classification Accuracy: {severity_accuracy * 100:.2f}%')

    run.log({
        "test_accuracy_disease": disease_accuracy,
        "test_accuracy_severity": severity_accuracy,
    })

    disease_precision = precision_score(all_disease_labels, all_disease_predictions, average='weighted', zero_division=1)
    severity_precision = precision_score(all_severity_labels, all_severity_predictions, average='weighted', zero_division=1)
    disease_recall = recall_score(all_disease_labels, all_disease_predictions, average='weighted', zero_division=1)
    severity_recall = recall_score(all_severity_labels, all_severity_predictions, average='weighted', zero_division=1)
    disease_f1 = f1_score(all_disease_labels, all_disease_predictions, average='weighted', zero_division=1)
    severity_f1 = f1_score(all_severity_labels, all_severity_predictions, average='weighted', zero_division=1)


    # Log metrics to wandb
    run.log({
        "test_precision_disease": disease_precision,
        "test_precision_severity": severity_precision,
        "test_recall_disease": disease_recall,
        "test_recall_severity": severity_recall,
        "test_f1_disease": disease_f1,
        "test_f1_severity": severity_f1,
    })

    info = {
        "run_name": run_name,
        "accuracy_disease": disease_accuracy,
        "accuracy_severity": severity_accuracy,
        "precision_disease": disease_precision,
        "precision_severity": severity_precision,
        "recall_disease": disease_recall,
        "recall_severity": severity_recall,
        "f1_disease": disease_f1,
        "f1_severity": severity_f1,
    }

    print("Metrics succesfuly logged to wandb!")


    
    # Save the trained model
    torch.save(model.state_dict(), "disease_severity_model.pth")
    run.save("disease_severity_model.pth")

    with open("model_info.txt", "w") as f:
        f.write(json.dumps(info, indent=4))

    wandb.join()


#### --------------------------------------------
#### freeze_training_disease_first.py

class DiseaseSeverityModel_Freeze_DiseaseFirst(nn.Module):
    def __init__(self, num_disease_classes, num_severity_levels):
        super(DiseaseSeverityModel_Freeze_DiseaseFirst, self).__init__()
        vgg19 = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True)
        for param in vgg19.features.parameters(): param.requires_grad = False
        self.features = vgg19.features
        self.avgpool = vgg19.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )
        self.disease_classifier = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_disease_classes),
        )
        self.severity_classifier = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_severity_levels),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        disease_output = self.disease_classifier(x)
        severity_output = self.severity_classifier(x)
        return disease_output, severity_output


def run_experiment_freeze_disease(args):
    def train(args, train_data, val_data, test_data, device, run):
        # Initialize the model
        model = args.get("model", DiseaseSeverityModel_Freeze_DiseaseFirst(num_disease_classes=4, num_severity_levels=5))
        model.to(device)
        run.watch(model)

        criterion = args.get("criterion", nn.CrossEntropyLoss())
        optimizer = args.get("optimizer", torch.optim.Adam(model.parameters(), lr=args.get("lr", 0.001)))

        # First train backbone + disease classification head
        for param in model.severity_classifier.parameters():
            param.requires_grad = False

        epochs = args.get("epochs", 10)
        best_val_loss = float('inf')  # Initialize the best validation loss

        early_stopping = EarlyStopping(patience=4, verbose=True)

        # Train disease first
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            total_train_accuracy = 0
            for step, batch in enumerate(train_data):
                images = batch[0].to(device)
                disease_labels = batch[1].to(device)

                optimizer.zero_grad()
                disease_output, _ = model(images)
                disease_loss = criterion(disease_output, torch.argmax(disease_labels, dim=1))
                disease_loss.backward()
                optimizer.step()
                running_loss += disease_loss.item()

                # Calculate accuracy
                logits = disease_output.detach().cpu().numpy()
                label_ids = disease_labels.to('cpu').numpy()
                total_train_accuracy += flat_accuracy(logits, label_ids)

            avg_train_loss = running_loss / len(train_data)
            avg_train_accuracy = total_train_accuracy / len(train_data)
            print(
                f"Epoch {epoch + 1}/{epochs}, Training Loss - Disease: {avg_train_loss}, Training Accuracy - Disease: {avg_train_accuracy}")

            run.log({
                "train_loss_disease": avg_train_loss,
                "train_accuracy_disease": avg_train_accuracy,
            })

            # Validation
            model.eval()
            val_loss = 0.0
            total_val_accuracy = 0
            with torch.no_grad():
                for batch in val_data:
                    images = batch[0].to(device)
                    disease_labels = batch[1].to(device)
                    disease_output, _ = model(images)
                    disease_loss = criterion(disease_output, torch.argmax(disease_labels, dim=1))
                    val_loss += disease_loss.item()

                    # Calculate accuracy
                    logits = disease_output.detach().cpu().numpy()
                    label_ids = disease_labels.to('cpu').numpy()
                    total_val_accuracy += flat_accuracy(logits, label_ids)

            avg_val_loss = val_loss / len(val_data)
            avg_val_accuracy = total_val_accuracy / len(val_data)
            print(
                f"Epoch {epoch + 1}/{epochs}, Validation Loss - Disease: {avg_val_loss}, Validation Accuracy - Disease: {avg_val_accuracy}")

            # Log metrics to wandb
            run.log({
                "val_loss_disease": avg_val_loss,
                "val_accuracy_disease": avg_val_accuracy,
            })

            # Check if validation loss has improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "disease_first_severity_model_frozen.pth")  # Save the model
                run.save("disease_first_severity_model_frozen.pth")

            # Early stopping
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # Train severity second

        # Freeze all the layers except the severity classifier
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.avgpool.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = False
        for param in model.disease_classifier.parameters():
            param.requires_grad = False

        # Unfreeze the severity classifier
        for param in model.severity_classifier.parameters():
            param.requires_grad = True

        criterion = args.get("criterion", nn.CrossEntropyLoss())
        optimizer = args.get("optimizer", torch.optim.Adam(model.parameters(), lr=args.get("lr", 0.001)))

        epochs = args.get("epochs", 10)
        best_val_loss = float('inf')  # Initialize the best validation loss

        early_stopping = EarlyStopping(patience=4, verbose=True)

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            total_train_accuracy = 0
            for step, batch in enumerate(train_data):
                images = batch[0].to(device)
                severity_labels = batch[2].to(device)

                optimizer.zero_grad()
                _, severity_output = model(images)
                severity_loss = criterion(severity_output, torch.argmax(severity_labels, dim=1))
                severity_loss.backward()
                optimizer.step()
                running_loss += severity_loss.item()

                # Calculate accuracy
                logits = severity_output.detach().cpu().numpy()
                label_ids = severity_labels.to('cpu').numpy()
                total_train_accuracy += flat_accuracy(logits, label_ids)

            avg_train_loss = running_loss / len(train_data)
            avg_train_accuracy = total_train_accuracy / len(train_data)
            print(
                f"Epoch {epoch + 1}/{epochs}, Training Loss - Severity: {avg_train_loss}, Training Accuracy - Severity: {avg_train_accuracy}")

            run.log({
                "train_loss_severity": avg_train_loss,
                "train_accuracy_severity": avg_train_accuracy,
            })

            # Validation
            model.eval()
            val_loss = 0.0
            total_val_accuracy = 0
            with torch.no_grad():
                for batch in val_data:
                    images = batch[0].to(device)
                    severity_labels = batch[2].to(device)
                    _, severity_output = model(images)
                    severity_loss = criterion(severity_output, torch.argmax(severity_labels, dim=1))
                    val_loss += severity_loss.item()

                    # Calculate accuracy
                    logits = severity_output.detach().cpu().numpy()
                    label_ids = severity_labels.to('cpu').numpy()
                    total_val_accuracy += flat_accuracy(logits, label_ids)

            avg_val_loss = val_loss / len(val_data)
            avg_val_accuracy = total_val_accuracy / len(val_data)
            print(
                f"Epoch {epoch + 1}/{epochs}, Validation Loss - Severity: {avg_val_loss}, Validation Accuracy - Severity: {avg_val_accuracy}")

            # Log metrics to wandb
            run.log({
                "val_loss_severity": avg_val_loss,
                "val_accuracy_severity": avg_val_accuracy,
            })

            # Check if validation loss has improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "disease_first_severity_model_frozen.pth")  # Save the model
                run.save("disease_first_severity_model_frozen.pth")

            # Early stopping
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print("")
        print("Training complete!")

        return model


    # ------------------------------
    # ----- Data Preparation -------
    # ------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transformations = args.transforms if hasattr(args, 'transforms') else transforms.Compose([])

    name = args.name if hasattr(args, 'name') else "freeze_training_disease_first"

    full_dataset = DiaMOSDataset(csv_file=args.csv_file if hasattr(args, 'csv_file') else 'diaMOSPlant.csv',
                                 img_dir=args.img_dir if hasattr(args,
                                                                 'img_dir') else '/kaggle/input/diamos-plant-dataset/Pear/leaves',
                                 data_path=args.data_path if hasattr(args,
                                                                     'data_path') else '/kaggle/input/diamos-plant-dataset/Pear/',
                                 transform=transformations)

    train_size = args.train_size if hasattr(args, 'train_size') else int(0.6 * len(full_dataset))    
    
    augmentation = args.augmentation if hasattr(args, 'augmentation') else False
    # augment_operations = args.augment_operations if hasattr(args, 'augment_operations') else []
    augment_target_size_factor = args.augment_target_size_factor if hasattr(args, 'augment_target_size_factor') else 1
    augment_save_dir = args.augment_save_dir if hasattr(args, 'augment_save_dir') else '/kaggle/working/augmented/'
    
    val_size = args.val_size if hasattr(args, 'val_size') else int(0.2 * len(full_dataset))
    test_size = args.test_size if hasattr(args, 'test_size') else len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    if augmentation:
        print("Augmenting dataset...")
        print("Length of train dataset before augmentation: ", len(train_dataset))
        train_dataset.dataset.augment = True
        train_dataset.dataset.save_dir = augment_save_dir
        train_dataset.dataset.augment_dataset(len(train_dataset) * augment_target_size_factor, augment_save_dir)
        print("Length of train dataset after augmentation: ", len(train_dataset))

    batch_size = args.batch_size if hasattr(args, 'batch_size') else 16
    num_workers = args.num_workers if hasattr(args, 'num_workers') else 2

    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            pin_memory=True)
    val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # ------------------------------
    # ----- Model Training ---------
    # ------------------------------

    args_dict = vars(args)
    project_name = args.project_name if hasattr(args, 'project_name') else "plant_disease_detection"
    run_name = name + "_" + animal_version_name(project_name)
    print("Project Name:", project_name)
    run = wandb.init(name=run_name, reinit=True, entity="plant_disease_detection", project=project_name)
    serializable_args_dict = {k: v for k, v in args_dict.items() if isinstance(v, (int, float, str, bool, list, dict, tuple, set))}
    run.config.update(serializable_args_dict)
    model = train(args_dict, train_data, val_data, test_data, device, run)

    if augmentation:
        print("Removing augmented images...")
        os.rmdir(augment_save_dir)
        print("Augmented images removed!")

    # ------------------------------
    # ----- Model Evaluation -------
    # ------------------------------

    # Evaluation loop (for validation or test data)
    model.eval()

    # Add code for evaluation here (using validation or test data)
    total_disease_correct = 0
    total_severity_correct = 0
    total_samples = 0

    # # Initialize lists to store correctly classified images (to print them in the next cell)
    # correct_images = []
    # correct_disease_labels = []
    # correct_severity_labels = []
    # correct_disease_predictions = []
    # correct_severity_predictions = []

    all_disease_labels = []
    all_disease_predictions = []
    all_severity_labels = []
    all_severity_predictions = []

    with torch.no_grad():
        for images, disease_labels, severity_labels in test_data:
            images, disease_labels, severity_labels = images.to(device), disease_labels.to(device), severity_labels.to(device)
            disease_output, severity_output = model(images)
            
            # Disease classification
            _, disease_predicted = torch.max(disease_output, 1)
            disease_correct = (disease_predicted == torch.argmax(disease_labels, dim=1))
            total_disease_correct += disease_correct.sum().item()
            
            # Severity classification
            _, severity_predicted = torch.max(severity_output, 1)
            severity_correct = (severity_predicted == torch.argmax(severity_labels, dim=1))
            total_severity_correct += severity_correct.sum().item()
            
            total_samples += images.size(0)
            
            # Find indices of correctly classified images
            # correct_indices = (disease_correct & severity_correct).nonzero().squeeze()

            # # Find indices of correctly classified images
            # correct_indices = (disease_correct & severity_correct).nonzero().squeeze()

            # Check if correct_indices is a scalar
            # if correct_indices.dim() == 0:
            #     correct_indices = [correct_indices.item()]

            # # Append correctly classified images and their labels/predictions
            # correct_images.extend(images[correct_indices].cpu().numpy())
            # correct_disease_labels.extend(disease_labels[correct_indices].cpu().numpy())
            # correct_severity_labels.extend(severity_labels[correct_indices].cpu().numpy())
            # correct_disease_predictions.extend(disease_predicted[correct_indices].cpu().numpy())
            # correct_severity_predictions.extend(severity_predicted[correct_indices].cpu().numpy())

            # Append all disease labels and predictions
            # Append all disease labels and predictions
            all_disease_labels.extend(torch.argmax(disease_labels, dim=1).cpu().numpy())
            all_disease_predictions.extend(disease_predicted.cpu().numpy())
            all_severity_labels.extend(torch.argmax(severity_labels, dim=1).cpu().numpy())
            all_severity_predictions.extend(severity_predicted.cpu().numpy())

    disease_accuracy = total_disease_correct / total_samples
    severity_accuracy = total_severity_correct / total_samples
    print(f'Disease Classification Accuracy: {disease_accuracy * 100:.2f}%')
    print(f'Severity Classification Accuracy: {severity_accuracy * 100:.2f}%')

    run.log({
        "test_accuracy_disease": disease_accuracy,
        "test_accuracy_severity": severity_accuracy,
    })

    disease_precision = precision_score(all_disease_labels, all_disease_predictions, average='weighted', zero_division=1)
    severity_precision = precision_score(all_severity_labels, all_severity_predictions, average='weighted', zero_division=1)
    disease_recall = recall_score(all_disease_labels, all_disease_predictions, average='weighted', zero_division=1)
    severity_recall = recall_score(all_severity_labels, all_severity_predictions, average='weighted', zero_division=1)
    disease_f1 = f1_score(all_disease_labels, all_disease_predictions, average='weighted', zero_division=1)
    severity_f1 = f1_score(all_severity_labels, all_severity_predictions, average='weighted', zero_division=1)


    # Log metrics to wandb
    run.log({
        "test_precision_disease": disease_precision,
        "test_precision_severity": severity_precision,
        "test_recall_disease": disease_recall,
        "test_recall_severity": severity_recall,
        "test_f1_disease": disease_f1,
        "test_f1_severity": severity_f1,
    })

    info = {
        "run_name": run_name,
        "accuracy_disease": disease_accuracy,
        "accuracy_severity": severity_accuracy,
        "precision_disease": disease_precision,
        "precision_severity": severity_precision,
        "recall_disease": disease_recall,
        "recall_severity": severity_recall,
        "f1_disease": disease_f1,
        "f1_severity": severity_f1,
    }

    print("Metrics succesfuly logged to wandb!")



    # Save the trained model
    torch.save(model.state_dict(), "disease_first_severity_model_frozen.pth")
    run.save("disease_first_severity_model_frozen.pth")

    with open("model_info.txt", "w") as f:
        f.write(json.dumps(info, indent=4))

    wandb.join()



#### --------------------------------------------
#### freeze_training_severity_first.py


class DiseaseSeverityModel_Freeze_SeverityFirst(nn.Module):
    def __init__(self, num_disease_classes, num_severity_levels):
        super(DiseaseSeverityModel_Freeze_SeverityFirst, self).__init__()
        vgg19 = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True)
        for param in vgg19.features.parameters(): param.requires_grad = False
        self.features = vgg19.features
        self.avgpool = vgg19.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )
        self.disease_classifier = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_disease_classes),
        )
        self.severity_classifier = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_severity_levels),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        disease_output = self.disease_classifier(x)
        severity_output = self.severity_classifier(x)
        return disease_output, severity_output


def run_experiment_freeze_severity(args):
    def train(args, train_data, val_data, test_data, device, run):
        # Initialize the model
        model = args.get("model", DiseaseSeverityModel_Freeze_SeverityFirst(num_disease_classes=4, num_severity_levels=5))
        model.to(device)
        run.watch(model)

        criterion = args.get("criterion", nn.CrossEntropyLoss())
        optimizer = args.get("optimizer", torch.optim.Adam(model.parameters(), lr=args.get("lr", 0.001)))

        # First train backbone + severity classification head
        for param in model.disease_classifier.parameters():
            param.requires_grad = False

        epochs = args.get("epochs", 10)
        best_val_loss = float('inf')  # Initialize the best validation loss

        early_stopping = EarlyStopping(patience=4, verbose=True)

        # Train severity first
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            total_train_accuracy = 0
            for step, batch in enumerate(train_data):
                images = batch[0].to(device)
                severity_labels = batch[2].to(device)

                optimizer.zero_grad()
                _, severity_output = model(images)
                severity_loss = criterion(severity_output, torch.argmax(severity_labels, dim=1))
                severity_loss.backward()
                optimizer.step()
                running_loss += severity_loss.item()

                # Calculate accuracy
                logits = severity_output.detach().cpu().numpy()
                label_ids = severity_labels.to('cpu').numpy()
                total_train_accuracy += flat_accuracy(logits, label_ids)

            avg_train_loss = running_loss / len(train_data)
            avg_train_accuracy = total_train_accuracy / len(train_data)
            print(
                f"Epoch {epoch + 1}/{epochs}, Training Loss - Severity: {avg_train_loss}, Training Accuracy - Severity: {avg_train_accuracy}")

            run.log({
                "train_loss_severity": avg_train_loss,
                "train_accuracy_severity": avg_train_accuracy,
            })

            # Validation
            model.eval()
            val_loss = 0.0
            total_val_accuracy = 0
            with torch.no_grad():
                for batch in val_data:
                    images = batch[0].to(device)
                    severity_labels = batch[2].to(device)
                    _, severity_output = model(images)
                    severity_loss = criterion(severity_output, torch.argmax(severity_labels, dim=1))
                    val_loss += severity_loss.item()

                    # Calculate accuracy
                    logits = severity_output.detach().cpu().numpy()
                    label_ids = severity_labels.to('cpu').numpy()
                    total_val_accuracy += flat_accuracy(logits, label_ids)

            avg_val_loss = val_loss / len(val_data)
            avg_val_accuracy = total_val_accuracy / len(val_data)
            print(
                f"Epoch {epoch + 1}/{epochs}, Validation Loss - Severity: {avg_val_loss}, Validation Accuracy - Severity: {avg_val_accuracy}")

            # Log metrics to wandb
            run.log({
                "val_loss_severity": avg_val_loss,
                "val_accuracy_severity": avg_val_accuracy,
            })

            # Check if validation loss has improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "disease_severity_first_model_frozen.pth")  # Save the model
                run.save("disease_severity_first_model_frozen.pth")

            # Early stopping
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # Train disease second

        # Freeze all the layers except the disease classifier
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.avgpool.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = False
        for param in model.severity_classifier.parameters():
            param.requires_grad = False

        # Unfreeze the disease classifier
        for param in model.disease_classifier.parameters():
            param.requires_grad = True

        criterion = args.get("criterion", nn.CrossEntropyLoss())
        optimizer = args.get("optimizer", torch.optim.Adam(model.parameters(), lr=args.get("lr", 0.001)))

        epochs = args.get("epochs", 10)
        best_val_loss = float('inf')  # Initialize the best validation loss

        early_stopping = EarlyStopping(patience=4, verbose=True)

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            total_train_accuracy = 0
            for step, batch in enumerate(train_data):
                images = batch[0].to(device)
                disease_labels = batch[1].to(device)

                optimizer.zero_grad()
                disease_output, _ = model(images)
                disease_loss = criterion(disease_output, torch.argmax(disease_labels, dim=1))
                disease_loss.backward()
                optimizer.step()
                running_loss += disease_loss.item()

                # Calculate accuracy
                logits = disease_output.detach().cpu().numpy()
                label_ids = disease_labels.to('cpu').numpy()
                total_train_accuracy += flat_accuracy(logits, label_ids)

            avg_train_loss = running_loss / len(train_data)
            avg_train_accuracy = total_train_accuracy / len(train_data)
            print(
                f"Epoch {epoch + 1}/{epochs}, Training Loss - Disease: {avg_train_loss}, Training Accuracy - Disease: {avg_train_accuracy}")

            run.log({
                "train_loss_disease": avg_train_loss,
                "train_accuracy_disease": avg_train_accuracy,
            })

            # Validation
            model.eval()
            val_loss = 0.0
            total_val_accuracy = 0
            with torch.no_grad():
                for batch in val_data:
                    images = batch[0].to(device)
                    disease_labels = batch[1].to(device)
                    disease_output, _ = model(images)
                    disease_loss = criterion(disease_output, torch.argmax(disease_labels, dim=1))
                    val_loss += disease_loss.item()

                    # Calculate accuracy
                    logits = disease_output.detach().cpu().numpy()
                    label_ids = disease_labels.to('cpu').numpy()
                    total_val_accuracy += flat_accuracy(logits, label_ids)

            avg_val_loss = val_loss / len(val_data)
            avg_val_accuracy = total_val_accuracy / len(val_data)
            print(
                f"Epoch {epoch + 1}/{epochs}, Validation Loss - Disease: {avg_val_loss}, Validation Accuracy - Disease: {avg_val_accuracy}")

            # Log metrics to wandb
            run.log({
                "val_loss_disease": avg_val_loss,
                "val_accuracy_disease": avg_val_accuracy,
            })

            # Check if validation loss has improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "disease_severity_first_model_frozen.pth")  # Save the model
                run.save("disease_severity_first_model_frozen.pth")

            # Early stopping
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print("")
        print("Training complete!")

        return model


    # ------------------------------
    # ----- Data Preparation -------
    # ------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transformations = args.transforms if hasattr(args, 'transforms') else transforms.Compose([])

    name = args.name if hasattr(args, 'name') else "freeze_training_severity_first"

    full_dataset = DiaMOSDataset(csv_file=args.csv_file if hasattr(args, 'csv_file') else 'diaMOSPlant.csv',
                                 img_dir=args.img_dir if hasattr(args,
                                                                 'img_dir') else '/kaggle/input/diamos-plant-dataset/Pear/leaves',
                                 data_path=args.data_path if hasattr(args,
                                                                     'data_path') else '/kaggle/input/diamos-plant-dataset/Pear/',
                                 transform=transformations)

    train_size = args.train_size if hasattr(args, 'train_size') else int(0.6 * len(full_dataset))
    val_size = args.val_size if hasattr(args, 'val_size') else int(0.2 * len(full_dataset))
    test_size = args.test_size if hasattr(args, 'test_size') else len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    
    augmentation = args.augmentation if hasattr(args, 'augmentation') else False
    # augment_operations = args.augment_operations if hasattr(args, 'augment_operations') else []
    augment_target_size_factor = args.augment_target_size_factor if hasattr(args, 'augment_target_size_factor') else 1
    augment_save_dir = args.augment_save_dir if hasattr(args, 'augment_save_dir') else '/kaggle/working/augmented/'

    if augmentation:
        print("Augmenting dataset...")
        print("Length of train dataset before augmentation: ", len(train_dataset))
        train_dataset.dataset.augment = True
        train_dataset.dataset.save_dir = augment_save_dir
        train_dataset.dataset.augment_dataset(len(train_dataset) * augment_target_size_factor, augment_save_dir)
        print("Length of train dataset after augmentation: ", len(train_dataset))

    
    batch_size = args.batch_size if hasattr(args, 'batch_size') else 16
    num_workers = args.num_workers if hasattr(args, 'num_workers') else 2

    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            pin_memory=True)
    val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    if augmentation:
        print("Removing augmented images...")
        os.rmdir(augment_save_dir)
        print("Augmented images removed!")

    # ------------------------------
    # ----- Model Training ---------
    # ------------------------------

    args_dict = vars(args)
    project_name = args.project_name if hasattr(args, 'project_name') else "plant_disease_detection"
    run_name = name + "_" + animal_version_name(project_name)
    print("Project Name:", project_name)
    run = wandb.init(name=run_name, reinit=True, entity="plant_disease_detection", project=project_name)
    serializable_args_dict = {k: v for k, v in args_dict.items() if isinstance(v, (int, float, str, bool, list, dict, tuple, set))}
    run.config.update(serializable_args_dict)
    model = train(args_dict, train_data, val_data, test_data, device, run)

    # ------------------------------
    # ----- Model Evaluation -------
    # ------------------------------

    # Evaluation loop (for validation or test data)
    model.eval()

    # Add code for evaluation here (using validation or test data)
    total_disease_correct = 0
    total_severity_correct = 0
    total_samples = 0

    # # Initialize lists to store correctly classified images (to print them in the next cell)
    # correct_images = []
    # correct_disease_labels = []
    # correct_severity_labels = []
    # correct_disease_predictions = []
    # correct_severity_predictions = []

    all_disease_labels = []
    all_disease_predictions = []
    all_severity_labels = []
    all_severity_predictions = []

    with torch.no_grad():
        for images, disease_labels, severity_labels in test_data:
            images, disease_labels, severity_labels = images.to(device), disease_labels.to(device), severity_labels.to(device)
            disease_output, severity_output = model(images)

            # Disease classification
            _, disease_predicted = torch.max(disease_output, 1)
            disease_correct = (disease_predicted == torch.argmax(disease_labels, dim=1))
            total_disease_correct += disease_correct.sum().item()
            
            # Severity classification
            _, severity_predicted = torch.max(severity_output, 1)
            severity_correct = (severity_predicted == torch.argmax(severity_labels, dim=1))
            total_severity_correct += severity_correct.sum().item()
            
            total_samples += images.size(0)

            # # Find indices of correctly classified images
            # correct_indices = (disease_correct & severity_correct).nonzero().squeeze()

            # # Check if correct_indices is a scalar
            # if correct_indices.dim() == 0:
            #     correct_indices = [correct_indices.item()]

            # # Append correctly classified images and their labels/predictions
            # correct_images.extend(images[correct_indices].cpu().numpy())
            # correct_disease_labels.extend(disease_labels[correct_indices].cpu().numpy())
            # correct_severity_labels.extend(severity_labels[correct_indices].cpu().numpy())
            # correct_disease_predictions.extend(disease_predicted[correct_indices].cpu().numpy())
            # correct_severity_predictions.extend(severity_predicted[correct_indices].cpu().numpy())

            # Append all disease labels and predictions
            # Append all disease labels and predictions
            all_disease_labels.extend(torch.argmax(disease_labels, dim=1).cpu().numpy())
            all_disease_predictions.extend(disease_predicted.cpu().numpy())
            all_severity_labels.extend(torch.argmax(severity_labels, dim=1).cpu().numpy())
            all_severity_predictions.extend(severity_predicted.cpu().numpy())

    disease_accuracy = total_disease_correct / total_samples
    severity_accuracy = total_severity_correct / total_samples
    print(f'Disease Classification Accuracy: {disease_accuracy * 100:.2f}%')
    print(f'Severity Classification Accuracy: {severity_accuracy * 100:.2f}%')

    run.log({
        "test_accuracy_disease": disease_accuracy,
        "test_accuracy_severity": severity_accuracy,
    })

    disease_precision = precision_score(all_disease_labels, all_disease_predictions, average='weighted', zero_division=1)
    severity_precision = precision_score(all_severity_labels, all_severity_predictions, average='weighted', zero_division=1)
    disease_recall = recall_score(all_disease_labels, all_disease_predictions, average='weighted', zero_division=1)
    severity_recall = recall_score(all_severity_labels, all_severity_predictions, average='weighted', zero_division=1)
    disease_f1 = f1_score(all_disease_labels, all_disease_predictions, average='weighted', zero_division=1)
    severity_f1 = f1_score(all_severity_labels, all_severity_predictions, average='weighted', zero_division=1)


    # Log metrics to wandb
    run.log({
        "test_precision_disease": disease_precision,
        "test_precision_severity": severity_precision,
        "test_recall_disease": disease_recall,
        "test_recall_severity": severity_recall,
        "test_f1_disease": disease_f1,
        "test_f1_severity": severity_f1,
    })

    info = {
        "run_name": run_name,
        "accuracy_disease": disease_accuracy,
        "accuracy_severity": severity_accuracy,
        "precision_disease": disease_precision,
        "precision_severity": severity_precision,
        "recall_disease": disease_recall,
        "recall_severity": severity_recall,
        "f1_disease": disease_f1,
        "f1_severity": severity_f1,
    }

    print("Metrics succesfuly logged to wandb!")



    # Save the trained model
    torch.save(model.state_dict(), "disease_severity_first_model_frozen.pth")
    run.save("disease_severity_first_model_frozen.pth")

    with open("model_info.txt", "w") as f:
        f.write(json.dumps(info, indent=4))

    wandb.join()

#### --------------------------------------------
#### main.py

# EXPERIMENT1 = SimpleNamespace(
#     type = "divergent_heads",
#     transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ]),
#     img_dir = 'data\\Pear\\leaves\\',
#     data_path = 'data\\Pear\\',
#     name = "divergent_heads",
#     project_name = "Teo Runs",
#     epochs = 25,
# )

# EXPERIMENT2 = SimpleNamespace(
#     type = "divergent_heads",
#     transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor()
#     ]),
#     img_dir = 'data\\Pear\\leaves\\',
#     data_path = 'data\\Pear\\',
#     name = "divergent_heads_no_normalization",
#     project_name = "Teo Runs",
#     epochs = 25,
# )

# EXPERIMENT3 = SimpleNamespace(
#     type = "combined",
#     transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ]),
#     img_dir = 'data\\Pear\\leaves\\',
#     data_path = 'data\\Pear\\',
#     name = 'combined',
#     project_name = "Teo Runs",
#     epochs = 25,
# )

# EXPERIMENT4 = SimpleNamespace(
#     type = "combined",
#     transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#     ]),
#     img_dir = 'data\\Pear\\leaves\\',
#     data_path = 'data\\Pear\\',
#     name = 'combined_no_normalization',
#     project_name = "Teo Runs",
#     epochs = 25,
# )

# EXPERIMENT5 = SimpleNamespace(
#     type = "divergent_heads",
#     transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ]),
#     img_dir = 'data\\Pear\\leaves\\',
#     data_path = 'data\\Pear\\',
#     name = 'divergent_heads_resnet',
#     project_name = "Teo Runs",
#     epochs = 25,
#     model = DiseaseSeverityModelResNet50(num_disease_classes=4, num_severity_levels=5)
# )

# EXPERIMENT6 = SimpleNamespace(
#     type = "combined",
#     transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ]),
#     img_dir = 'data\\Pear\\leaves\\',
#     data_path = 'data\\Pear\\',
#     name = 'combined_resnet',
#     project_name = "Teo Runs",
#     epochs = 25,
#     model = DiseaseSeverityModel_CombinedResNet50(20)
# )

# EXPERIMENT_COMBINED = SimpleNamespace(
#     type = "combined",
#     transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ]),
#     # img_dir = 'data\\Pear\\leaves\\',
#     # data_path = 'data\\Pear\\',
#     name = 'combined',
#     project_name = "experiments",
#     epochs = 25,
# )

# EXPERIMENT_DIVERGENT = SimpleNamespace(
#     type = "divergent_heads",
#     transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ]),
#     # img_dir = 'data\\Pear\\leaves\\',
#     # data_path = 'data\\Pear\\',
#     name = "divergent_heads",
#     project_name = "experiments",
#     epochs = 25,
# )

# EXPERIMENT_SEPARATE = SimpleNamespace(
#     type = "separate",
#     transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ]),
#     # img_dir = 'data\\Pear\\leaves\\',
#     # data_path = 'data\\Pear\\',
#     name = "separate",
#     project_name = "experiments",
#     epochs = 25,
# )

# EXPERIMENT_FREEZE_DISEASE = SimpleNamespace(
#     type = "freeze_disease",
#     transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ]),
#     # img_dir = 'data\\Pear\\leaves\\',
#     # data_path = 'data\\Pear\\',
#     name = "freeze_disease",
#     project_name = "experiments",
#     epochs = 25,
# )

# EXPERIMENT_FREEZE_SEVERITY = SimpleNamespace(
#     type = "freeze_severity",
#     transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ]),
#     # img_dir = 'data\\Pear\\leaves\\',
#     # data_path = 'data\\Pear\\',
#     name = "freeze_severity",
#     project_name = "experiments",
#     epochs = 25,
# )

# EXPERIMENT_15X_AUGMENTED_FREEZE_D = SimpleNamespace(
#     type = "freeze_disease",
#     transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ]),
#     # img_dir = 'data\\Pear\\leaves\\',
#     # data_path = 'data\\Pear\\',
#     name = "1.5x_augmented_freeze_disease",
#     project_name = "experiments",
#     epochs = 25,
#     augmentation = True,
#     augment_target_size_factor = 1.5,

# )

# EXPERIMENT_20X_AUGMENTED_FREEZE_D = SimpleNamespace(
#     type = "freeze_disease",
#     transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ]),
#     # img_dir = 'data\\Pear\\leaves\\',
#     # data_path = 'data\\Pear\\',
#     name = "2x_augmented_freeze_disease",
#     project_name = "experiments",
#     epochs = 25,
#     augmentation = True,
#     augment_target_size_factor = 2,
#     augment_save_dir = '/kaggle/working/augmented2/',
# )

# EXPERIMENT_30X_AUGMENTED_FREEZE_D = SimpleNamespace(
#     type = "freeze_disease",
#     transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ]),
#     # img_dir = 'data\\Pear\\leaves\\',
#     # data_path = 'data\\Pear\\',
#     name = "3x_augmented_freeze_disease",
#     project_name = "experiments",
#     epochs = 25,
#     augmentation = True,
#     augment_target_size_factor = 3,
#     augment_save_dir = '/kaggle/working/augmented3/',
# )

# EXPERIMENT_15X_AUGMENTED_FREEZE_S = SimpleNamespace(
#     type = "freeze_severity",
#     transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ]),
#     # img_dir = 'data\\Pear\\leaves\\',
#     # data_path = 'data\\Pear\\',
#     name = "1.5x_augmented_freeze_severity",
#     project_name = "experiments",
#     epochs = 25,
#     augmentation = True,
#     augment_target_size_factor = 1.5,
# )

# EXPERIMENT_20X_AUGMENTED_FREEZE_S = SimpleNamespace(
#     type = "freeze_severity",
#     transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ]),
#     # img_dir = 'data\\Pear\\leaves\\',
#     # data_path = 'data\\Pear\\',
#     name = "2x_augmented_freeze_severity",
#     project_name = "experiments",
#     epochs = 25,
#     augmentation = True,
#     augment_target_size_factor = 2,
#     augment_save_dir = '/kaggle/working/augmented2/',
# )

# EXPERIMENT_30X_AUGMENTED_FREEZE_S = SimpleNamespace(
#     type = "freeze_severity",
#     transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ]),
#     # img_dir = 'data\\Pear\\leaves\\',
#     # data_path = 'data\\Pear\\',
#     name = "3x_augmented_freeze_severity",
#     project_name = "experiments",
#     epochs = 25,
#     augmentation = True,
#     augment_target_size_factor = 3,
#     augment_save_dir = '/kaggle/working/augmented3/',
# )

# EXPERIMENT_FREEZE_DISEASE = SimpleNamespace(
#     type = "freeze_disease",
#     transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ]),
#     # img_dir = 'data\\Pear\\leaves\\',
#     # data_path = 'data\\Pear\\',
#     name = "freeze_disease",
#     project_name = "experiments",
#     epochs = 25,
# )

# EXPERIMENT_FREEZE_SEVERITY = SimpleNamespace(
#     type = "freeze_severity",
#     transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ]),
#     # img_dir = 'data\\Pear\\leaves\\',
#     # data_path = 'data\\Pear\\',
#     name = "freeze_severity",
#     project_name = "experiments",
#     epochs = 25,
# )

# EXPERIMENT_15X_SEPARATE = SimpleNamespace(
#     type = "freeze_severity",
#     transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ]),
#     img_dir = 'data\\Pear\\leaves\\',
#     data_path = 'data\\Pear\\',
#     name = "1.5x_augmented_freeze_severity",
#     project_name = "Teo Test Runs",
#     epochs = 25,
#     augmentation = True,
#     augment_target_size_factor = 1.5,
# )

VGG = SimpleNamespace(
    type = "separate",
    transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    name = "vgg_separate",
    project_name = "experiments",
    epochs = 25,
    disease_model = DiseaseModel(num_disease_classes=4),
    severity_model = SeverityModel(num_severity_levels=5),
)

RESNET = SimpleNamespace(
    type = "separate",
    transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    name = "resnet_separate",
    project_name = "experiments",
    epochs = 25,
    disease_model = DiseaseModelResNet(num_disease_classes=4),
    severity_model = SeverityModelResNet(num_severity_levels=5),
)

DENSE = SimpleNamespace(
    type = "separate",
    transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    name = "dense_separate",
    project_name = "experiments",
    epochs = 25,
    disease_model = DiseaseModelDenseNet(num_disease_classes=4),
    severity_model = SeverityModelDenseNet(num_severity_levels=5),
)

INCEPTION = SimpleNamespace(
    type = "separate",
    transforms = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    name = "inception_separate",
    project_name = "experiments",
    epochs = 25,
    disease_model = DiseaseModelInception(num_disease_classes=4),
    severity_model = SeverityModelInception(num_severity_levels=5),
)

MOBILE = SimpleNamespace(
    type = "separate",
    transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    name = "mobile_separate",
    project_name = "experiments",
    epochs = 25,
    disease_model = DiseaseModelMobile(num_disease_classes=4),
    severity_model = SeverityModelMobile(num_severity_levels=5),
)

EXPERIMENTS = [
    # VGG,
    # RESNET,
    DENSE, 
    INCEPTION,
    MOBILE,
]


def run_experiments(experiments):
    for experiment in experiments:
        try:
            if experiment.type == "separate":
                run_experiment_separate(experiment)
            elif experiment.type == "combined":
                run_experiment_combined(experiment)
            elif experiment.type == "divergent_heads":
                run_experiment_divergent(experiment)
            elif experiment.type == 'freeze_disease':
                run_experiment_freeze_disease(experiment)
            elif experiment.type == 'freeze_severity':
                run_experiment_freeze_disease(experiment)
            else:
                print("Invalid type")
                sys.exit(1)
            print("Experiment completed: ", str(experiment))
        except Exception as e:
            print(f"Error running experiment: {e}")
            print(traceback.format_exc())
            continue


wandb.login(key="a8acb651e87c4dca872eeb0bdedcfccf93ab7171")
run_experiments(EXPERIMENTS)
wandb.finish()