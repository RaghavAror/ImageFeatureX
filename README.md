# VQA Fine-tuning on Amazon Product Images using PaliGemma

This repository contains code and training scripts to fine-tune Google’s [`paligemma-3b-pt-224`](https://huggingface.co/google/paligemma-3b-pt-224) model for **Visual Question Answering (VQA)**, applied to the [Amazon ML Challenge 2023](https://www.kaggle.com/competitions/amazon-ml-challenge-2023/data). The objective is to extract product attributes (like weight, dimensions, etc.) directly from product images using multimodal learning.

---

## Objective of model

Given a product image and a question like “What is the weight?”, the model must predict the corresponding attribute **(e.g., “34 gram”)**. The dataset contains image URLs, product group IDs, and labeled entity types/values. During test time, only the image and question are provided.

Our task is to train a model that:
- Learns from `train.csv` (image + question ➝ answer),
- Predicts on `test.csv` (image + question ➝ unknown),
- Formats the predictions precisely (as described below),
- Passes the `sanity.py` format checker.

---

## Dataset Description

| Column        | Description |
|---------------|-------------|
| `index`       | Unique identifier for each sample |
| `image_link`  | Public URL of the product image |
| `group_id`    | Categorical identifier for the product |
| `entity_name` | The attribute to be predicted (e.g., "item_weight") |
| `entity_value`| **Target column (not present in test)** — e.g., “34 gram” |

💡 `entity_value` is your **ground truth** during training and the **prediction** during inference.

---


---

## Model Used

| Component     | Description |
|---------------|-------------|
| **Base Model** | [`google/paligemma-3b-pt-224`](https://huggingface.co/google/paligemma-3b-pt-224) |
| **Tokenizer & Vision Processor** | `PaliGemmaProcessor` |
| **Quantization** | 4-bit using `bitsandbytes` |
| **Fine-Tuning Strategy** | LoRA (PEFT) + Supervised Fine-Tuning (SFTTrainer) |
| **Frameworks** | `transformers`, `trl`, `peft`, `accelerate` |

---


---

## Code Walkthrough

### 1. Load the Dataset

- Load `train.csv`, filter only useful columns: `image_link`, `entity_name`, `entity_value`.
- Build image-text pairs: prompt is `"What is the <entity_name>?"`, answer is `entity_value`.

### 2. Preprocess & Tokenize

- Use `PaliGemmaProcessor` to prepare vision + text inputs.
- Resize images to 224x224, normalize using vision processor.

### 3. Define LoRA PEFT Model

- Use `peft.get_peft_model()` on `AutoModelForVision2Seq`.
- Apply LoRA on transformer layers to reduce training cost.

### 4. Train with `SFTTrainer`

- Provide `train_dataset`, `collate_fn`, `model`, and tokenizer to `SFTTrainer`.
- Log metrics every few steps.
- Save checkpoints for later inference.

### 5. Inference & Prediction Formatting

- For each test image, construct input: `prompt = "What is the <entity_name>?"`
- Decode model output to plain text (e.g., `"2 gram"`)
- Format predictions into a CSV matching `sample_test_out.csv`





