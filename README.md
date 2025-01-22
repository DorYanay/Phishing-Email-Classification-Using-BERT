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
2. **Fine-Tuning**: Train the model on the phishing classification dataset.
3. **Evaluation**: Test the model on a separate validation set.

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

