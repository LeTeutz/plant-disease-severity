import json
import time

import torch
import wandb
from sklearn.metrics import classification_report
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms

from utils import DiaMOSDataset, format_time, flat_accuracy, EarlyStopping, animal_version_name


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
            run.log({"loss": loss.item(), "val_loss": avg_val_loss})
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            
            logits_disease = disease_output.detach().cpu().numpy()
            logits_severity = severity_output.detach().cpu().numpy()
            
            disease_label_ids = disease_labels.to('cpu').numpy()
            severity_label_ids = severity_labels.to('cpu').numpy()
            
            total_train_accuracy_disease += flat_accuracy(logits_disease, disease_label_ids)
            total_train_accuracy_severity += flat_accuracy(logits_severity, severity_label_ids)


        avg_train_accuracy_disease = total_train_accuracy_disease / len(train_data)
        print(" Train Accuracy - Disease: {0:.2f}".format(avg_train_accuracy_disease))
            
        avg_train_accuracy_severity = total_train_accuracy_severity / len(train_data)
        print(" Train Accuracy - Severity: {0:.2f}".format(avg_train_accuracy_severity))
        
        run.log({
            "loss": running_loss / len(train_data),
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
                disease_output, severity_output = model(images)
                
            disease_loss = criterion(disease_output, torch.argmax(disease_labels, dim=1))
            severity_loss = criterion(severity_output, torch.argmax(severity_labels, dim=1))
            val_loss += (disease_loss + severity_loss).item()

            logits_disease = disease_output.detach().cpu().numpy()
            logits_severity = severity_output.detach().cpu().numpy()
            
            disease_label_ids = disease_labels.to('cpu').numpy()
            severity_label_ids = severity_labels.to('cpu').numpy()
            
            total_val_accuracy_disease += flat_accuracy(logits_disease, disease_label_ids)
            total_val_accuracy_severity += flat_accuracy(logits_severity, severity_label_ids)


        avg_val_loss = val_loss / len(val_data)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_data)}, Validation Loss: {avg_val_loss}")
        
        avg_val_accuracy_disease = total_val_accuracy_disease / len(val_data)
        avg_val_accuracy_severity = total_val_accuracy_severity / len(val_data)

        print(" Validation Accuracy - Disease: {0:.2f}".format(avg_val_accuracy_disease))
        print(" Validation Accuracy - Severity: {0:.2f}".format(avg_val_accuracy_severity))

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
        
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))


        scheduler.step()


    # Load the best model
    model.load_state_dict(torch.load("disease_severity_model.pth"))

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    return model


def run_experiment(args):
    
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
    run.config.update(args_dict)
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

    # Compute precision, recall, and accuracy per label for disease classification
    disease_report = classification_report(all_disease_labels, all_disease_predictions, output_dict=True)

    # Compute precision, recall, and accuracy per label for severity classification
    severity_report = classification_report(all_severity_labels, all_severity_predictions, output_dict=True)

    print(f'Disease Classification Accuracy: {disease_accuracy * 100:.2f}%')
    print(f'Severity Classification Accuracy: {severity_accuracy * 100:.2f}%')

    # Log metrics to wandb
    run.log({
        "test_accuracy_disease": disease_accuracy,
        "test_accuracy_severity": severity_accuracy,
        "disease_report": disease_report,
        "severity_report": severity_report,
    })

    info = {
        "run_name": run_name,
        "hyperparameters": args_dict,
        "accuracy_disease": disease_accuracy,
        "accuracy_severity": severity_accuracy,
        "disease_report": disease_report,
        "severity_report": severity_report,
    }
    
    # Save the trained model
    torch.save(model.state_dict(), "disease_severity_model.pth")
    run.save("disease_severity_model.pth")

    with open("model_info.txt", "w") as f:
        f.write(json.dumps(info, indent=4))

    wandb.join()