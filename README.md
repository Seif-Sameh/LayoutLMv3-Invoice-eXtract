# LayoutLMv3-Invoice Extraction: Fine-Tuning for Invoice Understanding

## 1. **Project Overview**
This project focuses on fine-tuning **LayoutLMv3** for document image understanding, specifically for automating data extraction from financial documents such as invoices. The goal is to extract key invoice entities, including invoice numbers, dates, total amounts, and line items, using a transformer-based approach that processes both text and layout information.

---

## 2. **Project Objectives**
- Fine-tune LayoutLMv3 on the **SERIO** dataset, which contains annotated invoices.
- Improve the model's ability to recognize structured data from financial documents.
- Automate key aspects of financial processing by accurately extracting and labeling relevant invoice data.

---

## 3. **Environment Setup**

### Prerequisites
To run this project, you'll need the following libraries:
- **os, glob, shutil**: For file and directory handling.
- **PIL (Python Imaging Library)**: For image processing.
- **cv2 (OpenCV)**: For image manipulation and visualization.
- **torch, torchvision**: For working with PyTorch deep learning models.
- **transformers**: For using LayoutLMv3 from the HuggingFace model hub.
- **scikit-learn**: For evaluation metrics such as accuracy, precision, recall, and F1-score.
- **matplotlib**: For visualizing bounding boxes and images.
- **tqdm**: For progress tracking during training.

### Installation

```bash
pip install torch torchvision transformers scikit-learn matplotlib opencv-python tqdm Pillow
```

---

## 4. **Data Preparation**

### 4.1 Dataset Overview
The dataset used is the **SERIO** dataset, which contains annotated invoices. Each invoice includes bounding box coordinates for key elements such as:
- **Invoice Number**
- **Invoice Date**
- **Total Amount**
- **Line Items**

### 4.2 Preprocessing Steps
- **Image Processing**: Convert images to appropriate dimensions and format.
- **Bounding Box Extraction**: Extract bounding box coordinates for each entity in the invoice.
- **Tokenization**: Convert the invoice text into tokens using the `LayoutLMv3Tokenizer`.
- **Annotation Formatting**: Prepare the annotations in a format that can be used for token classification.

Tokenization code snippet:

```python
from transformers import LayoutLMv3Tokenizer

tokenizer = LayoutLMv3Tokenizer.from_pretrained("mp-02/layoutlmv3-large-cord2")

# Example of converting text and layout to tokens
tokens = self.tokenizer(
    words,
    boxes=bboxes,
    truncation=True,
    padding='max_length',
    max_length=self.max_length,
    is_split_into_words=True,
    return_tensors="pt"
)
```

### 4.3 Splitting the Dataset
The dataset is divided into three sets:
- **Training Set**: 626 example of the data for training the model.
- **Validation Set**: 173 example of the data for model validation.
- **Test Set**: 174 example of the data for final model evaluation.

---

## 5. **Model Architecture**

### LayoutLMv3 Overview
**LayoutLMv3** is a transformer model designed for document image understanding. It processes both textual content and the layout information of the document (spatial arrangement of text) to extract key entities. In this project, we fine-tune LayoutLMv3 to perform **Named Entity Recognition (NER)** on invoice data.

### Model Components
- **Tokenizer**: `LayoutLMv3Tokenizer` tokenizes both the textual and layout features.
- **Model**: `LayoutLMv3ForTokenClassification` includes a token classification head, which assigns labels (such as `INVOICE_NUMBER`, `DATE`, `TOTAL_AMOUNT`) to each token.

### Model Initialization

```python
from transformers import LayoutLMv3ForTokenClassification

model = LayoutLMv3ForTokenClassification.from_pretrained(
    "mp-02/layoutlmv3-large-cord2",
    num_labels=5,
    hidden_dropout_prob=0.2
)
```

---

## 6. **Training the Model**

### 6.1 Training Loop
The model is trained using the standard PyTorch training loop. For each batch, the following steps are taken:
- **Forward Pass**: The invoice data is passed through the model to obtain predictions.
- **Loss Calculation**: The loss is computed between the predicted labels and the true labels using a token classification loss function (cross-entropy).
- **Backpropagation**: The model parameters are updated based on the computed loss to minimize error.
- **Checkpointing**: Model checkpoints are saved to avoid losing progress during training.

### Model Training

```python
import torch
from transformers import Trainer, TrainingArguments

class_weights = torch.tensor([5.0, 5.0, 5.0, 5.0, 1.0])  # Adjusted weights

for idx, param in enumerate(model.parameters()):
    param.requires_grad = idx >= 8

def custom_loss_func(logits, labels):
    loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)  # Use class weights here
    return loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

model.loss_fct = custom_loss_func

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=100,
    num_train_epochs=40,
    learning_rate=1e-5,
    report_to='wandb',
    run_name='layoutlmv3-training',
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,  # Use the entire training dataset
    eval_dataset=val_set,    # Use the entire validation dataset
    compute_metrics=compute_metrics
)

trainer.train()
```

---

## 7. **Evaluation**

### 7.1 Evaluation Metrics
The model is evaluated using several metrics:
- **Accuracy**: Measures the percentage of correctly predicted labels.
- **Precision**: Measures the proportion of positive identifications that were actually correct.
- **Recall**: Measures the proportion of actual positives that were identified correctly.
- **F1-Score**: Harmonic mean of precision and recall, providing a balanced measure of the model's performance.

### 7.2 Validation Set Evaluation
The validation set is used during training to monitor the model's performance and prevent overfitting.

### 7.3 Test Set Evaluation
After training, the model is evaluated on a held-out test set to assess its real-world performance.

---

### 8. **Model Results**

After training the LayoutLMv3 model on the SERIO dataset, we evaluated its performance on the test set. The evaluation involved extracting key invoice elements, including invoice numbers, dates, and totals. Here are the key metrics and results:

#### 8.1 Evaluation Metrics
We measured the model's performance using the following metrics:
- **Accuracy**: The percentage of correct entity predictions.
- **Precision**: The proportion of correctly predicted entities among all predicted entities.
- **Recall**: The proportion of actual entities that were correctly predicted.
- **F1-Score**: The harmonic mean of precision and recall, providing a balanced measure of the model’s performance.

**Evaluation Results:**
- **Accuracy**: 96.2%
- **Precision**: 95.4%
- **Recall**: 96.2%
- **F1-Score**: 95.8%

---

#### 8.2 Output Generation

The final step of the model's prediction pipeline involves generating structured data outputs in a **JSON** format. The predictions extracted from the test set include bounding boxes and labels for relevant entities such as:
- **Company Name**
- **Address**
- **Date**
- **Total Amount**

These extracted entities are formatted into a structured JSON file for further use or integration with external systems.

Example of the JSON output structure:

```json

[
    {
        "x0" : 29,
        "y0": 220, 
        "x2": 616,
        "y2": 258,
        "line": "(KJ1) 273500-U S/B CONFECTIONERY KING'S",
        "label": "S-COMPANY"
    },{

        "x0" : 29,
        "yo": 258,
        "x2": 574,
        "y2": 295,
        "line": "JAYA, KELANA S$6/3, JALAN 20-A1, NO. ",
        "label": "S-ADDRESS"
    },{
        "x0": 29,
        "yo": 303,
        "x2": 72,
        "y2" : 330,
        "line": "473",
        "label": "S-ADDRESS"
    }
]

```

The above JSON output structure reflects the model's ability to extract key information and organize it in a consumable format.

---

## 9. **Conclusion**

This project successfully fine-tunes the **LayoutLMv3** model for extracting key information from invoices. The model demonstrates strong performance in automating the extraction of structured data from financial documents, providing a useful tool for automating business processes related to invoice management.

---

## 10. **Future Work**
- **Further Fine-tuning**: Experiment with different learning rates and hyperparameters for improved accuracy.
- **Additional Data**: Incorporate more diverse financial documents to improve generalization.
- **Integration**: Deploy the model in a production environment for real-time invoice processing.
