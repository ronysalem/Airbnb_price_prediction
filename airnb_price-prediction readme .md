
## Notebook Overview

This Jupyter notebook addresses a problem of predicting prices based on properties. It combines both text and image data to make two predictions: property type and price. The notebook explores various machine learning models and strategies to achieve accurate predictions.

---

## Setup Instructions

### Packages and Requirements

To run this Jupyter notebook, you'll need the following packages and requirements:

- Python (3.x recommended)
- Jupyter Notebook
- NumPy
- Pandas
- Seaborn
- Matplotlib
- Scikit-Learn
- TensorFlow
- HoloViews
- NLTK
- Googletrans
- Keras
- Scikit-Optimize
- Python-Levenshtein
- Pandas-Profiling
- patool
- tqdm

You can install the required packages using pip, for example:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn tensorflow nltk keras holoviews googletrans scikit-optimize python-Levenshtein pandas-profiling patool tqdm
```

### How to Use

1. Download the dataset from [this link](https://github.com/CISC-873/Information-2021/releases/download/data/a4-5.zip) and extract it using a tool like patool.

2. Open the Jupyter notebook in your Jupyter Notebook environment.

3. Run each cell of the notebook sequentially to follow the code and analysis.

4. Explore the different trials and models to understand the process of predicting property type and price based on the provided data.

---

## Data Preparation

The notebook uses data from the 'train_xy.csv' and 'test_x.csv' files for training and testing. It includes the following data preparation steps:

- Converting labels to categorical values.
- Handling missing values.
- Loading and processing images.
- Preprocessing text data, including translation to English.
- Building a vocabulary from the training set.

---

## Model Training

The notebook explores various machine learning models and architectures. Some of the key models used include:

1. Trial 1: Combining convolutional layers for images and simple average embedding for text.
2. Trial 2: Combining convolutional layers with dropout for images and LSTM with dropout for text.
3. Trial 3: Combining convolutional layers with dropout for images and GRU with dropout for text.
4. Trial 4: Using Bidirectional LSTM for text and convolutional layers for images.
5. Trial 5: Using simple average embedding for text to predict price.
6. Trial 6: Using convolutional layers with dropout for images to predict price.

Each trial is evaluated for training and validation scores. Details on the model architecture, loss functions, and training processes are included.

---

## Results and Observations

The results of each trial are summarized as follows:

- Training score
- Validation score
- Kaggle public score (if applicable)

The observations and insights gained from each trial are included, helping to understand the performance and limitations of each model.

---

## Questions

The notebook addresses the following questions:

- Is a fully-connected model suitable for sequential data and image data? Why or why not?
- What are gradient vanishing and gradient explosion, and how do GRU/LSTM mitigate these problems?
- What is multi-objective/multi-task learning, and how is it used in this assignment?
- What is multi-modality learning, and how is it used in this assignment?
- What are the differences among XGBoost, LightGBM, and CatBoost?
