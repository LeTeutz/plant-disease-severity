import sys
import wandb
import separate_classifiers
from divergent_heads_classifier import DiseaseSeverityModelVGG, DiseaseSeverityModelResNet50, train_divergent, run_experiment_divergent
from combined_classifier import DiseaseSeverityModel_CombinedVGG, DiseaseSeverityModel_CombinedResNet50, train_combined, run_experiment_combined
from torchvision import transforms
from types import SimpleNamespace



EXPERIMENT1 = SimpleNamespace(
    classifier = "divergent_heads",
    transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    img_dir = 'data\\Pear\\leaves\\',
    data_path = 'data\\Pear\\',
    name = "divergent_heads",
    project_name = "Teo Test Runs",
    epochs = 25,
)

EXPERIMENT2 = SimpleNamespace(
    classifier = "divergent_heads",
    transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]),
    img_dir = 'data\\Pear\\leaves\\',
    data_path = 'data\\Pear\\',
    name = "divergent_heads_no_normalization",
    project_name = "Teo Test Runs",
    epochs = 25,
)

EXPERIMENT3 = SimpleNamespace(
    classifier = "combined",
    transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    img_dir = 'data\\Pear\\leaves\\',
    data_path = 'data\\Pear\\',
    name = 'combined',
    project_name = "Teo Test Runs",
    epochs = 25,
)

EXPERIMENT4 = SimpleNamespace(
    classifier = "combined",
    transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]),
    img_dir = 'data\\Pear\\leaves\\',
    data_path = 'data\\Pear\\',
    name = 'combined_no_normalization',
    project_name = "Teo Test Runs",
    epochs = 25,
)

EXPERIMENT5 = SimpleNamespace(
    classifier = "divergent_heads",
    transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    img_dir = 'data\\Pear\\leaves\\',
    data_path = 'data\\Pear\\',
    name = 'divergent_heads_resnet',
    project_name = "Teo Test Runs",
    epochs = 25,
    model = DiseaseSeverityModelResNet50(num_disease_classes=4, num_severity_levels=5)
)

EXPERIMENT6 = SimpleNamespace(
    classifier = "combined",
    transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    img_dir = 'data\\Pear\\leaves\\',
    data_path = 'data\\Pear\\',
    name = 'combined_resnet',
    project_name = "Teo Test Runs",
    epochs = 25,
    model = DiseaseSeverityModel_CombinedResNet50(20)
)

EXPERIMENTS = [
    # EXPERIMENT1,
    # EXPERIMENT2,
    # EXPERIMENT3,
    # EXPERIMENT4,
    EXPERIMENT5,
    EXPERIMENT6,
]


def run_experiments(experiments):
    for experiment in experiments:
        if experiment.classifier == "separate":
            separate_classifiers.run_experiment(experiment)
        elif experiment.classifier == "combined":
            run_experiment_combined(experiment)
        elif experiment.classifier == "divergent_heads":
            run_experiment_divergent(experiment)
        else:
            print("Invalid classifier type")
            sys.exit(1)


if __name__ == '__main__':
    wandb.login(key="a8acb651e87c4dca872eeb0bdedcfccf93ab7171")
    run_experiments(EXPERIMENTS)
    wandb.finish()