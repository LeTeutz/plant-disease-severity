import sys
import wandb
import separate_classifiers
import divergent_heads_classifier
import combined_classifier
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
    project_name = "Teo Test Runs",
    epochs = 25
)

EXPERIMENT2 = SimpleNamespace(
    classifier = "combined",
    transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    img_dir = 'data\\Pear\\leaves\\',
    data_path = 'data\\Pear\\',
    project_name = "Teo Test Runs",
    epochs = 25
)

EXPERIMENTS = [
    EXPERIMENT1,
    EXPERIMENT2
]


def run_experiments(experiments):
    for experiment in experiments:
        if experiment.classifier == "separate":
            separate_classifiers.run_experiment(experiment)
        elif experiment.classifier == "combined":
            combined_classifier.run_experiment(experiment)
        elif experiment.classifier == "divergent_heads":
            divergent_heads_classifier.run_experiment(experiment)
        else:
            print("Invalid classifier type")
            sys.exit(1)


if __name__ == '__main__':
    wandb.login(key="a8acb651e87c4dca872eeb0bdedcfccf93ab7171")
    run_experiments(EXPERIMENTS)
    wandb.finish()