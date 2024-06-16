import json
import time
import os
import torch
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms
from utils import DiaMOSDataset, format_time, flat_accuracy, EarlyStopping, animal_version_name
from efficientnet_pytorch import EfficientNet
from torchvision.models import densenet201
from torchvision.models import mobilenet_v2
from torchvision.models import inception_v3
from torchvision.models import resnet50

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


def run_experiment(args):
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
        print("Length of train dataset before augmentation: ", len(train_dataset.dataset))
        train_dataset.dataset.augment = True
        train_dataset.dataset.aug_dir = augment_save_dir
        train_dataset.dataset.augment_dataset(len(train_dataset.dataset) * augment_target_size_factor, augment_save_dir)
        print("Length of train dataset after augmentation: ", len(train_dataset.dataset))
        train_dataset = torch.utils.data.Subset(train_dataset.dataset, range(len(train_dataset.dataset)))

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

