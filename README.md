# Image Classification for City Dog Show

> A Python pipeline that applies pre-trained CNN models to verify dog registrations — benchmarking AlexNet, VGG, and ResNet across accuracy and runtime.

---

## Problem Statement

A city is hosting a citywide dog show, and the organizing committee needs to verify that every registered contestant has submitted a genuine dog photo. Participants upload an image along with biographical information about their dog, and the registration system tags images based on that information.

The challenge: some people may attempt to register pets that aren't actually dogs.

The goal is to use an already-developed Python image classifier to catch invalid registrations — and to determine which classification algorithm performs best for this task.

> **Note:** The classifier itself was provided. The engineering work was building the full Python pipeline around it — argument parsing, label extraction, classification, results adjustment, statistics, and formatted output.

---

## Objectives

1. Correctly identify which pet images are of **dogs** — even if the breed is misclassified — and which are **not dogs**.
2. Correctly classify the **breed** of dog for images that are of dogs.
3. Determine which CNN architecture — **ResNet**, **AlexNet**, or **VGG** — best achieves objectives 1 and 2.
4. Measure the **runtime** of each algorithm and assess whether a faster model provides a "good enough" result given the accuracy–speed trade-off.

---

## Models Compared

Three pre-trained CNN architectures (trained on ImageNet) were evaluated:

| Model | Characteristics |
|-------|----------------|
| **AlexNet** | Lightweight, fast, pioneering architecture (2012) |
| **VGG** | Deeper network, higher accuracy, slower inference |
| **ResNet** | Residual connections — strong accuracy with reasonable speed |

---

## Key Results

| Model | Dog Detection | Breed Classification | Runtime |
|-------|:---:|:---:|:---:|
| **VGG** | Highest | Highest | Slowest |
| **ResNet** | High | High | Moderate |
| **AlexNet** | Moderate | Lowest | Fastest |

- **VGG** achieved the highest overall accuracy for both dog detection and breed classification — the recommended choice when correctness is the priority.
- **ResNet** offered a strong accuracy–speed balance, making it a viable alternative when runtime is a constraint.
- **AlexNet** was the fastest but showed the lowest accuracy, particularly for breed classification.

> This reflects a classic ML trade-off: **accuracy vs. computational cost**. For a real-world registration system, VGG is the best model — but ResNet is a compelling "good enough" alternative if speed matters.

---

## Pipeline

The pipeline is modular, with each Python file handling one responsibility:

```
Image Folder
     │
     ▼
get_pet_labels.py           → { filename: [true_label] }
     │
     ▼
classify_images.py          → { filename: [true_label, predicted_label, match_flag] }
     │
     ▼
adjust_results4_isadog.py   → marks each image as dog or not-dog for both the true label and the predicted label
     │
     ▼
calculates_results_stats.py → accuracy metrics per model
     │
     ▼
print_results.py            → formatted console output
```

---

## Project Structure

```
image-classification-for-city-dog-show/
│
├── check_images.py                    # Main entry point — orchestrates the full pipeline
├── get_input_args.py                  # Parses CLI args: image folder, model arch, dog names file
├── get_pet_labels.py                  # Extracts true pet labels from image filenames
├── classify_images.py                 # Runs CNN classifier; stores predicted labels + match flag
├── adjust_results4_isadog.py          # Checks if the true and predicted labels are dogs (1 = dog, 0 = not a dog)
├── calculates_results_stats.py        # Computes accuracy statistics per model
├── print_results.py                   # Formats and prints results to console
│
├── classifier.py                      # Pre-trained CNN model wrapper (AlexNet / VGG / ResNet)
├── test_classifier.py                 # Sanity tests for the classifier function
├── print_functions_for_lab_checks.py  # Unit-style checks for pipeline functions
│
├── dognames.txt                       # Reference list of valid dog breed names
├── imagenet1000_clsid_to_human.txt    # ImageNet class ID → human-readable label mapping
│
├── pet_images/                        # Test dataset: standard pet images
├── uploaded_images/                   # Test dataset: custom uploaded images
│
├── alexnet_pet-images.txt             # AlexNet results on pet_images
├── alexnet_uploaded-images.txt        # AlexNet results on uploaded_images
├── resnet_pet-images.txt              # ResNet results on pet_images
├── resnet_uploaded-images.txt         # ResNet results on uploaded_images
├── vgg_pet-images.txt                 # VGG results on pet_images
├── vgg_uploaded-images.txt            # VGG results on uploaded_images
│
├── run_models_batch.sh                # Runs all 3 models on pet_images
└── run_models_batch_uploaded.sh       # Runs all 3 models on uploaded_images
```

---

## Usage

**Run a single model:**
```bash
python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
```

**Run all three models in batch:**
```bash
bash run_models_batch.sh
bash run_models_batch_uploaded.sh
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--dir` | Path to folder containing pet images | `pet_images/` |
| `--arch` | CNN architecture: `resnet`, `alexnet`, or `vgg` | `vgg` |
| `--dogfile` | Text file with valid dog breed names | `dognames.txt` |

---

## Skills Demonstrated

- **Python & modular software design** — clean separation of concerns across pipeline stages
- **CLI argument parsing** with `argparse`
- **Dictionary-based data structures** for efficient label storage and lookup
- **Pre-trained CNN evaluation** using PyTorch's `torchvision` models
- **Performance benchmarking** — timing each model's inference across datasets
- **Shell scripting** for batch model execution
- **Comparative model analysis** — structured reasoning about accuracy vs. runtime trade-offs

---

## Requirements

```
Python 3.x
PyTorch
torchvision
Pillow
```

---

## About

This project is part of the **AWS AI Scientist Nanodegree** by Udacity.
It was the first hands-on project in the program, focused on applying Python programming skills to real-world AI tasks using pre-trained deep learning models.

---

*Built with Python | Deep Learning via PyTorch | Udacity × AWS Nanodegree*
