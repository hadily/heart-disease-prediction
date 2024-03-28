# Health Monitor Project - Machine Learning Part

This repository contains the machine learning component of the Health Monitor project, which aims to predict the presence or absence of heart disease based on various patient attributes.

## Dataset Information

The dataset used for this project contains the following features:

* Patient Identification Number: Numeric
* Age: Numeric (in years)
* Gender: Binary (0: female, 1: male)
* Resting blood pressure: Numeric (94-200 mm Hg)
* Serum cholesterol: Numeric (126-564 mg/dL)
* Fasting blood sugar: Binary (0: false/ <= 120 mg/dL, 1: true/ > 120 mg/dL)
* Chest pain type: Nominal (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)
* Resting electrocardiogram results: Nominal (0: normal, 1: ST-T wave abnormality, 2: probable or definite left ventricular hypertrophy)
* Maximum heart rate achieved: Numeric (71-202)
* Exercise induced angina: Binary (0: no, 1: yes)
* Oldpeak (ST depression induced by exercise relative to rest): Numeric (0-6.2)
* Slope of the peak exercise ST segment: Nominal (1: upsloping, 2: flat, 3: downsloping)
* Number of major vessels: Numeric (0, 1, 2, 3)
* Classification (target): Binary (0: absence of heart disease, 1: presence of heart disease)

## Algorithms Used

The following machine learning algorithms were employed for heart disease prediction:

* Random Forest Classifier
* Logistic Regression
* Support Vector Classifier (SVC)
* Decision Tree Classifier
* KNeighbors Classifier
* Gaussian Naive Bayes

  
