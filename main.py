import sys
import wandb
import separate_classifiers
import divergent_heads_classifier
import combined_classifier
import freeze_training_disease_first
import freeze_training_severity_first
import traceback
from torchvision import transforms
from types import SimpleNamespace


EXPERIMENT1 = SimpleNamespace(
    type = "separate",
    transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    img_dir = 'data\\Pear\\leaves\\',
    data_path = 'data\\Pear\\',
    project_name = "Teo Test Runs",
    name = "2x_augmented_separate",
    epochs = 25,
    augmentation = True,
    augment_target_size_factor = 2,
    augment_save_dir = 'data\\Pear\\leaves\\augmented\\'
)




# EXPERIMENT1 = SimpleNamespace(
#     type = "divergent_heads",
#     transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ]),
#     img_dir = 'data\\Pear\\leaves\\',
#     data_path = 'data\\Pear\\',
#     project_name = "Teo Test Runs",
#     epochs = 25
# )


# EXPERIMENT2 = SimpleNamespace(
#     type = "combined",
#     transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ]),
#     img_dir = 'data\\Pear\\leaves\\',
#     data_path = 'data\\Pear\\',
#     project_name = "Teo Test Runs",
#     epochs = 25
# )


# EXPERIMENT_15X_SEPARATE = SimpleNamespace(
#     type = "separate",
#     transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ]),
#     img_dir = 'data\\Pear\\leaves\\',
#     data_path = 'data\\Pear\\',
#     name = "1.5x_augmented_separated",
#     project_name = "Teo Test Runs",
#     epochs = 25,
#     augmentation = True,
#     augment_target_size_factor = 1.05,
#     augment_save_dir = 'data\\Pear\\leaves\\augmented\\',
# )

# EXPERIMENTS = [
#     EXPERIMENT_15X_SEPARATE,
# ]



def run_experiments(experiments):
    for experiment in experiments:
        try:
            if experiment.type == "separate":
                separate_classifiers.run_experiment(experiment)
            elif experiment.type == "combined":
                combined_classifier.run_experiment(experiment)
            elif experiment.type == "divergent_heads":
                divergent_heads_classifier.run_experiment(experiment)
            elif experiment.type == 'freeze_disease':
                freeze_training_disease_first.run_experiment(experiment)
            elif experiment.type == 'freeze_severity':
                freeze_training_severity_first.run_experiment(experiment)
            else:
                print("Invalid classifier type")
                sys.exit(1)
            print("Experiment completed: ", str(experiment))
        except Exception as e:
            print(f"Error running experiment: {e}")
            print(traceback.format_exc())
            continue

if __name__ == '__main__':
    wandb.login(key="a8acb651e87c4dca872eeb0bdedcfccf93ab7171")
    run_experiments(EXPERIMENTS)
    wandb.finish()