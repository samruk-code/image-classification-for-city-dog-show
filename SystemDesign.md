# Requirements — Image Classification for City Dog Show

This document walks through the ML system design phases for this project. Since the project uses **pre-trained CNNs for inference only** (no model training) and runs as a **batch CLI pipeline** (no deployed service), each phase below is scoped to what is actually relevant here.

---

## 1. Clarify Requirements & Constraints

### Business context

A city is hosting a citywide dog show. Participants register by uploading a photo of their pet along with biographical information. Some participants may attempt to register pets that are **not dogs**. The organizing committee needs an automated check that flags invalid (non-dog) registrations.

### Functional requirements

- **FR-1:** Given a folder of pet images, determine for each image whether it contains a **dog** or **not a dog**.
- **FR-2:** For images that do contain dogs, identify the **breed**.
- **FR-3:** Compare three CNN architectures — **AlexNet**, **VGG**, and **ResNet** — on both tasks and report which performs best.
- **FR-4:** Measure and report the **runtime** of each architecture so accuracy can be weighed against computational cost.
- **FR-5:** Produce summary statistics per model run: % correct dogs, % correct non-dogs, % correct breeds, % label matches, and counts of images/dogs/non-dogs.

### Constraints

- **The classifier is provided as-is** (`classifier.py`). The engineering scope is the pipeline *around* it: argument parsing, label extraction, classification orchestration, dog/not-dog adjustment, statistics, and reporting.
- **No training or fine-tuning.** Models are `torchvision` weights pre-trained on ImageNet (1000 classes).
- **Ground truth comes from filenames.** True labels are derived from image filenames (e.g., `Basenji_00963.jpg` → `basenji`), so filenames must follow the `Breed_name_NNNNN.jpg` convention.
- **Dog identification depends on a reference list.** `dognames.txt` defines which labels count as dogs; both true and predicted labels are checked against it.
- **Batch, offline execution.** Runs as a CLI on a local machine (CPU inference); there is no latency SLA, no online service, and no concurrency requirement.
- **Small evaluation sets.** 40 curated pet images plus 4 custom uploaded images — sufficient for comparative benchmarking, not for statistically rigorous evaluation.

### Success criteria

1. 100% correct dog vs. not-dog identification (this is the mission-critical check for registration).
2. Highest achievable breed accuracy among the compared models.
3. A justified recommendation of one architecture, considering the accuracy–runtime trade-off.

---

## 2. Define the ML Problem Formally

The project decomposes into two stacked classification problems, both solved by a single ImageNet classifier plus post-processing:

### Task A — Binary classification (primary): *dog vs. not-dog*

- **Input:** an RGB image `x`.
- **Model output:** the top-1 ImageNet class label `ŷ ∈ {1000 ImageNet classes}`.
- **Decision rule:** `is_dog(ŷ) = 1` if `ŷ` (or any of its comma-separated synonym terms) appears in `dognames.txt`, else `0`.
- **Ground truth:** `is_dog(y)` where `y` is the label parsed from the filename.
- **Metrics:** `% correct dogs` (recall on dog images) and `% correct non-dogs` (recall on non-dog images). Both matter: a false negative rejects a legitimate contestant; a false positive lets a non-dog into the show.

### Task B — Fine-grained multi-class classification (secondary): *breed identification*

- **Input:** images where the true label is a dog.
- **Prediction is correct** when the true breed string is contained in the predicted label (ImageNet labels include synonyms, e.g. `"dalmatian, coach dog, carriage dog"`).
- **Metric:** `% correct breed` = correctly-classified breeds / number of dog images.

### Model comparison objective

Select `arch* = argmax over {alexnet, resnet, vgg}` of classification performance (Task A first, Task B second), tie-broken by runtime — i.e., prefer the faster model when accuracy is equivalent.

---

## 3. Data Pipeline & Feature Engineering

There is no learned feature engineering (models are pre-trained); the data work is **label extraction, image preprocessing, and results enrichment**, organized as a modular pipeline where each stage owns one responsibility and passes a shared results dictionary forward:

```
Image folder (pet_images/ or uploaded_images/)
      │
      ▼
get_pet_labels.py            → {filename: [true_label]}
      │                         • parses filename, lowercases, strips digits/underscores
      │                         • skips hidden files (leading ".")
      ▼
classify_images.py           → {filename: [true_label, predicted_label, match(0/1)]}
      │                         • calls classifier(image_path, model_name)
      │                         • lowercases/strips prediction; substring match vs. truth
      ▼
adjust_results4_isadog.py    → appends [is_dog(true), is_dog(predicted)] flags
      │                         • membership check against dognames.txt
      ▼
calculates_results_stats.py  → counts + percentage statistics per run
      │
      ▼
print_results.py             → formatted console report (+ optional misclassification lists)
```

### Image preprocessing (inside `classifier.py`)

Standard ImageNet inference transforms applied per image:

1. `Resize(256)` → `CenterCrop(224)` — fixed 224×224 input.
2. `ToTensor()` — scale to [0, 1].
3. `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` — ImageNet channel statistics.
4. `unsqueeze(0)` — add batch dimension (batch size 1), gradients disabled for inference.

### Datasets

| Dataset | Contents | Purpose |
|---|---|---|
| `pet_images/` | 40 images (30 dogs across ~20 breeds, 10 non-dog animals/objects) | Primary benchmark |
| `uploaded_images/` | 4 custom images (dog, flipped dog, cup, parrot) | Robustness spot-check on unseen uploads |

### Data quality rules

- Filenames must encode the true label (`Breed_name_NNNNN.jpg`).
- `dognames.txt` must cover every dog breed/synonym that can appear as a true or predicted label.
- All label comparison is done on lowercased, whitespace-stripped strings.

---

## 4. Model Selection & Training Strategy

**No training occurs in this project** — all three candidates are `torchvision` models with frozen ImageNet weights, used purely for inference. The relevant work is therefore **candidate selection and empirical comparison**:

### Candidates

| Model | `torchvision` weights | Character |
|---|---|---|
| AlexNet | `alexnet(pretrained=True)` | Shallow, fastest, 2012-era baseline |
| ResNet-18 | `resnet18(pretrained=True)` | Residual connections; accuracy/speed balance |
| VGG-16 | `vgg16(pretrained=True)` | Deep, most accurate, slowest |

### Evaluation protocol

1. Run the full pipeline once per architecture on the same dataset (`run_models_batch.sh` / `run_models_batch_uploaded.sh` automate all three, logging to `<arch>_<dataset>.txt`).
2. Record the four percentage metrics plus total runtime for each run.
3. Rank models by dog/non-dog correctness first, breed accuracy second, runtime as the trade-off axis.

### Results and selection

| Model | % Correct Dogs | % Correct Non-Dogs | % Correct Breed | Runtime |
|---|:---:|:---:|:---:|:---:|
| **VGG** | 100% | 100% | 93.3% | Slowest |
| **ResNet** | 100% | 90% | 90.0% | Moderate |
| **AlexNet** | 100% | 100% | 80.0% | Fastest (~8× vs. VGG) |

**Decision: VGG** — the only model with perfect dog/non-dog separation *and* the highest breed accuracy, which matches the success criteria. **AlexNet** is the documented fallback when throughput matters: it matches VGG on the mission-critical dog/not-dog task and trades breed accuracy for an ~8× speedup. ResNet is eliminated because it was the only model to admit a non-dog as a dog — the costliest error type for this use case.

---

## 5. Serving & Inference Architecture

Serving here is a **local batch CLI**, not an online service:

### Entry point

```bash
python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
```

| Argument | Description | Default |
|---|---|---|
| `--dir` | Folder of images to classify | `pet_images/` |
| `--arch` | `resnet` \| `alexnet` \| `vgg` | `vgg` |
| `--dogfile` | Reference list of valid dog names | `dognames.txt` |

### Execution model

- **Orchestration:** `check_images.py` wires the pipeline stages sequentially and times the full run (`time.time()` start→end, reported as hh:mm:ss).
- **Inference:** CPU, single image at a time (batch size 1), model in `eval()` mode with gradients disabled.
- **Batch mode:** shell scripts run all three architectures back-to-back and redirect each run's console report to a results text file for side-by-side comparison.
- **Output:** human-readable console report — per-run counts, the four percentage metrics, and (optionally) lists of misclassified dogs and misclassified breeds.

### Dependencies

```
Python 3.x, PyTorch, torchvision, Pillow
```

*(Out of scope by design: REST/API serving, GPU inference, batching/throughput optimization, containerization. If the committee productionized this, the pipeline would sit behind the registration upload endpoint, running the chosen VGG model per submitted image.)*

---

## 6. Monitoring & Iteration

There is no live deployment, so monitoring takes the form of **offline result auditing and comparative iteration**:

### Result auditing

- Each model×dataset run is persisted to a results file (`vgg_pet-images.txt`, `alexnet_uploaded-images.txt`, etc.), giving a durable record for comparing runs and reviewing regressions after any pipeline change.
- `print_results.py` supports flags to print the individual **misclassified dogs** and **misclassified breeds**, enabling per-image error analysis (e.g., ResNet's single non-dog false positive).
- `print_functions_for_lab_checks.py` and `test_classifier.py` provide sanity checks that each pipeline stage's dictionary structure and the classifier itself behave as expected.

### Robustness checks

- The `uploaded_images/` set acts as a holdout probing known failure modes: the same dog image horizontally flipped (orientation sensitivity), and confusable non-dogs (cup, parrot).

### Iteration paths

If the system needed to improve, the highest-leverage next steps would be:

1. **Expand the evaluation set** — 40 images is too small to distinguish 93% from 100% reliably; grow the benchmark before trusting the ranking further.
2. **Fine-tune on dog breeds** — ImageNet covers ~120 dog classes; fine-tuning on a dedicated breed dataset would lift the 93.3% breed accuracy.
3. **Harden label matching** — the substring-based label/breed matching and filename-derived ground truth are the most fragile links; a normalized breed taxonomy would reduce false mismatches.
4. **Batch/GPU inference** — if registration volume grows, batching images through the model would recover most of VGG's runtime penalty.
