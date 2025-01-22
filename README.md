# Email Phishing Classification using BERT

## Project Overview
This project aims to classify emails as either phishing or legitimate using a pre-trained BERT (Bidirectional Encoder Representations from Transformers) model. By leveraging state-of-the-art natural language processing (NLP) techniques, the model can analyze email text and identify phishing attempts with high accuracy.

---

## Features
- **Phishing Detection**: Identifies whether an email is a phishing attempt.
- **Pre-trained BERT Model**: Utilizes the power of BERT for contextual understanding of email content.
- **Custom Fine-Tuning**: Adapts BERT to the phishing classification dataset.
- **Data Preprocessing**: Cleans and tokenizes email text for optimal model performance.

---

### Requirements
The `requirements.txt` includes:
- `transformers`
- `torch`
- `scikit-learn`
- `pandas`
- `numpy`
- `tqdm`
- `matplotlib`
- `seaborn`

---

## Dataset
The project uses datasets containing labeled email samples from the following sources:
- [Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)
- [Phishing Emails](https://www.kaggle.com/datasets/subhajournal/phishingemails)

### Data Preprocessing
1. Remove unnecessary metadata.
2. Clean email text by removing HTML tags and special characters.
3. Tokenize the text using BERT tokenizer.
.

---

## Model Training
### Steps
1. **Load Pre-trained BERT**: Use a BERT base model (`bert-base-uncased`) from the `transformers` library.
2. **Fine-Tuning**: Train the model on the phishing classification dataset using the following training arguments:

   ```python
   training_args = TrainingArguments(
       output_dir='./results',
       evaluation_strategy='steps',
       learning_rate=3e-5,
       per_device_train_batch_size=batch_size,
       per_device_eval_batch_size=batch_size,
       num_train_epochs=1,
       weight_decay=0.01,
       logging_dir='./logs',
       logging_steps=100,
       save_steps=100,
       save_strategy='steps',
       load_best_model_at_end=True,
       metric_for_best_model='eval_loss',
       fp16=use_fp16,  # Dynamically determined mixed precision
       report_to='none'
   )
   ```

3. **Evaluation**: Test the model on a separate validation set.

---

## Adversarial Training
### Overview
To enhance the robustness of the BERT model, adversarial training is implemented by generating modified phishing email examples with subtle alterations to phishing keywords (e.g., replacing 'a' with '@', 'o' with '0'). This strategy tests the model's ability to detect phishing emails that attempt to evade detection.

### Process
1. Identify common phishing-related keywords (e.g., "bank", "account", "login").
2. Generate adversarial examples by slightly modifying the keywords within the email text.
3. Evaluate the model on these adversarial examples to measure its robustness.

### Code Snippet
The following function generates adversarial examples and evaluates the model:

```python
# Implement Adversarial Training for Robustness
def adversarial_training(dataset, model, tokenizer, num_adversarial_examples=1000):
    phishing_keywords = ["bank", "account", "password", "login", "verify", "security", "urgent", "immediate", "credit", "transaction"]

    adversarial_texts = []
    for i in range(num_adversarial_examples):
        original_text = dataset.texts[i]
        modified_text = original_text
        for keyword in phishing_keywords:
            if keyword in modified_text:
                modified_text = modified_text.replace(keyword, keyword.replace('a', '@').replace('o', '0').replace('e', 'ë').replace('i', 'ï'))
        adversarial_texts.append(modified_text)

    adversarial_labels = dataset.labels[:num_adversarial_examples]
    adversarial_dataset = EmailDataset(adversarial_texts, adversarial_labels)

    print("\nEvaluating on Adversarial Examples:")
    evaluate_model(trainer, adversarial_dataset)
```
---

## Evaluation Metrics
The model is evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

A detailed report and confusion matrix are generated after evaluation.

---

## Results
| Metric         | Score       |
|----------------|-------------|
| Accuracy       | 99.41%      |
| Precision      | 99.19%      |
| Recall         | 99.63%      |


---

## Visualization
Performance metrics and training loss/accuracy are visualized using Matplotlib.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments
- Hugging Face Transformers Library
- PyTorch Framework

