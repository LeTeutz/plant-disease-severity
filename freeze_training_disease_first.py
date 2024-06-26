import json
import os
import torch
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms

from utils import DiaMOSDataset, flat_accuracy, EarlyStopping, animal_version_name


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


def run_experiment(args):
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
        print("Length of train dataset before augmentation: ", len(train_dataset.dataset))
        train_dataset.dataset.augment = True
        train_dataset.dataset.aug_dir = augment_save_dir
        train_dataset.dataset.augment_dataset(len(train_dataset.dataset) * augment_target_size_factor, augment_save_dir)
        print("Length of train dataset after augmentation: ", len(train_dataset.dataset))
        train_dataset = torch.utils.data.Subset(train_dataset.dataset, range(len(train_dataset.dataset)))

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
