# 🐾 Image Classification for City Dog Show
 
> **Udacity AWS AI/ML Scientist Nanodegree — Project 1**  
> Evaluating CNN architectures (AlexNet, VGG, ResNet) to classify dog breeds and filter non-dog registrations for a citywide dog show.
 
---
 
## 📌 Project Overview
 
A city is hosting a dog show and needs a reliable way to verify that every contestant has submitted a genuine dog photo. Participants upload an image and provide biographical information — but some may try to register pets that aren't dogs.
 
This project applies **pre-trained Convolutional Neural Networks (CNNs)** to:
1. Determine whether a submitted image is of a dog or not.
2. Identify the **breed** of the dog when applicable.
3. **Benchmark three CNN architectures** — AlexNet, VGG, and ResNet — across accuracy and runtime to recommend the best solution.
 
The classifier itself was provided; the engineering challenge was building the full Python pipeline around it: argument parsing, label extraction, image classification, results adjustment, statistics calculation, and formatted output.
 
---
 
## 🎯 Objectives
 
| # | Objective |
|---|-----------|
| 1 | Correctly identify which images are of **dogs** vs. **not dogs** |
| 2 | Correctly identify the **breed** of dog for images classified as dogs |
| 3 | Determine which CNN architecture **best** meets objectives 1 & 2 |
| 4 | Assess the **accuracy–runtime trade-off** to see if a faster model gives "good enough" results |
 
---
 
## 🧠 Models Compared
 
Three pre-trained CNN architectures (trained on ImageNet) were evaluated:
 
- **AlexNet** — Lightweight, fast, pioneering architecture (2012)
- **VGG** — Deeper network, higher accuracy, slower inference
- **ResNet** — Residual connections, strong accuracy with reasonable speed
 
---
 
## 🗂️ Project Structure
 
```
image-classification-for-city-dog-show/
│
├── check_images.py                    # 🚀 Main entry point — orchestrates the full pipeline
├── get_input_args.py                  # Parses CLI args: image folder, model arch, dog names file
├── get_pet_labels.py                  # Extracts true pet labels from image filenames
├── classify_images.py                 # Runs CNN classifier; stores predicted labels + match flag
├── adjust_results4_isadog.py          # Labels each result: is the true/predicted label a dog?
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
├── run_models_batch.sh                # Shell script: runs all 3 models on pet_images
└── run_models_batch_uploaded.sh       # Shell script: runs all 3 models on uploaded_images
```
 
---
 
## ⚙️ How It Works
 
The pipeline is modular, with each Python file handling one responsibility:
 
```
Image Folder
     │
     ▼
get_pet_labels.py          → { filename: [true_label] }
     │
     ▼
classify_images.py         → { filename: [true_label, predicted_label, match_flag] }
     │
     ▼
adjust_results4_isadog.py  → appends is_dog flags for true + predicted labels
     │
     ▼
calculates_results_stats.py → accuracy metrics per model
     │
     ▼
print_results.py            → formatted console output
```
 
---
 
## 🚀 Usage
 
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
| `--dir`  | Path to folder containing pet images | `pet_images/` |
| `--arch` | CNN architecture: `resnet`, `alexnet`, or `vgg` | `vgg` |
| `--dogfile` | Text file with valid dog breed names | `dognames.txt` |
 
---
 
## 📊 Key Results & Findings
 
Results across all three architectures were logged to `.txt` files and are available in this repository. The key findings:
 
- **VGG** achieved the highest overall accuracy for both dog detection and breed classification — the best model for this task.
- **ResNet** offered a strong accuracy–speed balance, making it a viable alternative when runtime is constrained.
- **AlexNet** was the fastest but showed the lowest accuracy, particularly for breed classification.
 
> This illustrates a classic ML trade-off: **accuracy vs. computational cost**. For a real-world registration system where correctness matters, VGG is the recommended choice — but if speed is critical, ResNet provides a compelling "good enough" alternative.
 
---
 
## 🛠️ Skills Demonstrated
 
- **Python & modular software design** — clean separation of concerns across pipeline stages
- **CLI argument parsing** with `argparse`
- **Dictionary-based data structures** for efficient label storage and lookup
- **Pre-trained CNN evaluation** using PyTorch's `torchvision` models
- **Performance benchmarking** — timing each model's inference across datasets
- **Shell scripting** for batch model execution
- **Comparative model analysis** — structured reasoning about accuracy vs. runtime trade-offs
 
---
 
## 📦 Requirements
 
```bash
Python 3.x
PyTorch
torchvision
Pillow
```
 
---
 
## 🎓 About
 
This project is part of the **AWS AI & ML Scientist Nanodegree** by Udacity.  
It was the first hands-on project in the program, focused on applying Python programming skills to real-world AI tasks using pre-trained deep learning models.
 
---
 
*Built with Python 🐍 | Deep Learning via PyTorch 🔥 | Udacity × AWS Nanodegree 🎓*
