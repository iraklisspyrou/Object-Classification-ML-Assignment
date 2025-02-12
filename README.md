# Object Classification for Autonomous Driving

## Overview
This project focuses on object classification for autonomous driving using machine learning models. The dataset used is the **KITTI dataset**, from which objects such as **Cars, Pedestrians, Cyclists, and Trams** were extracted and classified. Several machine learning models were trained and evaluated to determine the most effective approach.

## Dataset
The **KITTI dataset** was used, which contains:
- **data_object_image_2.zip**: Left color images for object detection (~12GB)
- **data_object_label.zip**: Corresponding label files (~5MB)
- **7481 training images** and **testing images** for evaluation

## Methodology
1. **Preprocessing & Feature Extraction**
   - **Cropping**: Bounding boxes were used to extract object images
   - **HOG (Histogram of Oriented Gradients)**: Feature extraction to create meaningful representations
   - **Scaling**: Normalization applied to features
   - **Balancing Techniques**: SMOTE and undersampling
   - **Dimensionality Reduction**: PCA applied to optimize feature space

2. **Model Training & Evaluation**
   - Models used: **SVM, k-NN, Gaussian Naive Bayes (GNB), Random Forest (RF)**
   - Hyperparameter tuning performed using **k-fold cross-validation**
   - Evaluation metrics: **Classification report, confusion matrices, ROC curves**

## Results
- **SVM & k-NN**: Achieved the best overall performance
- **GNB & RF**: Struggled particularly in the **Cyclist and Tram** classes due to low representation
- **SMOTE Impact**:
  - **RF** improved significantly after applying SMOTE
  - **k-NN & GNB** suffered a performance drop after balancing
  - **SVM** remained stable regardless of SMOTE application

## Repository Structure
```
├── code/
│   ├── cropping.py       #Crop the raw images and keep only the classes "Car", "Pedestrian", "Cyclist", "Tram" 
│   ├── HOG_feature_extraction.py       #Extract HoG features
│   ├── SVM.ipynb  # SVM training
│   ├── KNN.ipynb  # KNN training
│   ├── GNB.ipynb  # GNB training
│   └── Random_Forests.ipynb  # RF training
│
│
├── results/
│   ├── classification_reports/
│   ├── confusion_matrices/
│   ├── roc_curves/
│
├── README.md  # Project Documentation
├── requirements.txt  # Dependencies
├── ML_report.pdf  # Detailed report with results and analysis
├── ML_presentation #Presentation of all the workflow and results
```


## References
- **KITTI Dataset**: [http://www.cvlibs.net/datasets/kitti/](http://www.cvlibs.net/datasets/kitti/)
- **Scikit-learn Documentation**: [https://scikit-learn.org/](https://scikit-learn.org/)



