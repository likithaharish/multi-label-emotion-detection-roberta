# Emotion Detection using RoBERTa

## 📌 Overview

This project focuses on detecting emotions from text using a deep learning model.
We used the RoBERTa transformer model to perform multi-label classification, meaning a single sentence can have multiple emotions.

The model is trained on a subset of the GoEmotions dataset and can take real-time user input to predict emotions.

---

## 👥 Team Members

* Jeevan M
* Likitha H
* Keerthi KM
* I Leela Vyshnavi
* N Lakshmi Prasanna
* Jai Balaji

---

## ⚙️ Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* Scikit-learn
* Pandas, NumPy

---

## 📂 Dataset

* GoEmotions dataset (Reddit comments)
* Each text can have multiple emotion labels

---

## 🚀 Project Workflow

* Data loading and cleaning
* Tokenization using RoBERTa tokenizer
* Model training with multi-label classification
* Evaluation using F1-score, Precision, Recall
* Threshold tuning for better predictions
* Real-time user input for testing

---

## ▶️ How to Run

1. Install dependencies:

```
pip install pandas numpy torch transformers scikit-learn
```

2. Run the script:

```
python main.py
```

3. After training, enter input:

```
Enter text: I am feeling really excited today
```

4. The model will output predicted emotions.

---

## 🧠 Example

Input:

```
I feel happy but also nervous
```

Output:

```
{ joy: 0.61 }
```

---

## 📈 Features

* Multi-label emotion detection
* Transformer-based model (RoBERTa)
* Threshold tuning for improved performance
* Interactive user input after training

---

## ⚠️ Notes

* Training may take time depending on hardware
* GPU/MPS support improves speed
* Make sure dataset file is available in the project folder

---

## 🔮 Future Scope

* Convert into a web application
* Improve model accuracy
* Add visualization/dashboard for results

---

## 📁 Project Structure (basic)

```
├── main.py
├── go_emotions_dataset.csv
├── emotion_roberta/
├── results/
└── README.md
```

---

## 🤝 Contribution

This is a group project. Contributions were done collaboratively by all team members.

---

## 🎓 Guidance

This project was carried out at Jain University under the guidance of Dr. P. Renukadevi, Assistant Professor (CSE - AIML).
