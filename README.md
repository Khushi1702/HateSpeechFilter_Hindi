# HateSpeechFilter_Hindi
A machine learning project aimed at detecting hate speech in Hindi social media comments using Natural Language Processing (NLP) and various classification algorithms like SVM, Logistic Regression, and Random Forest.

## Overview
HateSpeechFilter_Hindi is a project aimed at detecting hate speech in Hindi social media comments. By utilizing Natural Language Processing (NLP) techniques and various machine learning models, this project focuses on preprocessing Hindi text data and classifying it into different sentiment categories, including hate speech.

## Features
- Data collection from social media platforms
- Preprocessing steps such as normalization, tokenization, stop words and punctuation removal
- TF-IDF vectorization for feature extraction
- Implementation of machine learning models: Logistic Regression, Naive Bayes, Support Vector Machine, Random Forest, and Gradient Boosting
- Evaluation of models using accuracy scores and other performance metrics

## Motivation
The proliferation of hate speech on social media necessitates robust detection mechanisms. This project addresses the challenges specific to Hindi text data, contributing to safer online interactions within the Hindi-speaking community.

## Problem Definition
To develop an efficient hate speech detection system for Hindi comments that can handle linguistic intricacies and accurately categorize sentiments. The system aims to enhance understanding and interpretation of emotions in Hindi text, providing valuable insights for various applications.

## Objectives
1. Develop effective machine learning models for hate speech detection in Hindi.
2. Apply comprehensive preprocessing techniques for Hindi text.
3. Achieve high accuracy in classifying hate speech and other sentiments.
4. Address challenges unique to Hindi text data.
5. Provide interpretability and insights into hate speech detection results.

## Tools and Technologies
- **Python**: Programming language used for implementation.
- **Pandas and NLTK**: For data manipulation and NLP tasks.
- **Scikit-learn**: For implementing machine learning models.
- **Matplotlib**: For visualizing evaluation results.

## Data Collection and Preprocessing
- **Data Collection**: Gathering Hindi comments from social media, e-commerce websites, and forums.
- **Preprocessing Steps**:
  - Normalization, tokenization, and removal of stop words and punctuation.
  - TF-IDF vectorization to convert text into numerical features.

## Model Implementation
- **Logistic Regression**: For predicting probabilities and classification.
- **Naive Bayes**: For handling feature independence assumptions.
- **Support Vector Machine (SVM)**: For effective linear and non-linear classification.
- **Random Forest**: For robust classification with multiple decision trees.
- **Gradient Boosting**: For handling complex relationships in data.

## Evaluation Metrics
- **Accuracy**: Measure of correctly identified instances.
- **Precision, Recall, F1-Score**: To evaluate model performance.

## Results
| Model                    | Accuracy (%) |
|--------------------------|--------------|
| Logistic Regression      | 68.76        |
| Support Vector Machine   | 67.28        |
| Multinomial Naive Bayes  | 58.90        |
| Gradient Boosting        | 62.48        |
| Random Forest            | 63.53        |

## Conclusion
Preprocessing techniques significantly enhance the performance of hate speech detection models. Logistic Regression and SVM models demonstrate high accuracy in classifying sentiments in Hindi comments. Challenges include class imbalance and linguistic variations.

## Future Scope
- Explore advanced deep learning architectures.
- Incorporate domain-specific lexicons.
- Integrate multimodal features for comprehensive detection.

## References
1. Sharma, A., & Kaushal, R. (2023). Detecting Hate Speech in Hindi in Online Social Media. *Proc. of the 3rd International Conference on Intelligent Communication and Computational Techniques (ICCT)*, Jaipur, India.
2. Jadhav, I., Kanade, A., Waghmare, V., & Chaudhari, D. (2022). Hate and Offensive Speech Detection in Hindi Twitter Corpus.
3. Chopra, S., Sawhney, R., Mathur, P., & Shah, R.R. (2020). Hindi-English Hate Speech Detection: Author Profiling, Debiasing, and Practical Perspectives. *Proc. of the AAAI Conference on Artificial Intelligence*.
4. Sreelakshmi, K., Premjith, B., & Soman, K.P. (2020). Detection of Hate Speech Text in Hindi-English Code-mixed Data. *Procedia Computer Science*.
5. Vashistha, N., & Zubiaga, A. (2021). Online Multilingual Hate Speech Detection: Experimenting with Hindi and English Social Media. *Information*.

## List of Abbreviations
- **NLP**: Natural Language Processing
- **TF-IDF**: Term Frequency-Inverse Document Frequency
- **SVM**: Support Vector Machine
- **ROC**: Receiver Operating Characteristic
- **ML**: Machine Learning
- **NLL**: Negative Log-Likelihood
