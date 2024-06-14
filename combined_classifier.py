import json
import time

import numpy as np
import torch
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms

from utils import DiaMOSDataset_Cartesian, format_time, flat_accuracy, EarlyStopping, animal_version_name


def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]


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


def run_experiment(args):
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

    # Compute overall precision, recall, and F1-score
    disease_precision = precision_score(all_disease_labels, all_disease_predictions, average='weighted')
    severity_precision = precision_score(all_severity_labels, all_severity_predictions, average='weighted')
    disease_recall = recall_score(all_disease_labels, all_disease_predictions, average='weighted')
    severity_recall = recall_score(all_severity_labels, all_severity_predictions, average='weighted')
    disease_f1 = f1_score(all_disease_labels, all_disease_predictions, average='weighted')
    severity_f1 = f1_score(all_severity_labels, all_severity_predictions, average='weighted')


    # Log metrics to wandb
    run.log({
        "test_accuracy_disease": disease_accuracy,
        "test_accuracy_severity": severity_accuracy,
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

    # Save the trained model
    torch.save(model.state_dict(), "disease_severity_model.pth")
    run.save("disease_severity_model.pth")

    with open("model_info.txt", "w") as f:
        f.write(json.dumps(info, indent=4))

    wandb.join()
