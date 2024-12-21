# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 19:15:25 2024

@author: irakl
"""

import os
import cv2
import csv

# Ορισμός διαδρομών
mostly_visible_objects_path = "C:\\Machine_Learning_Assignment\\cropped_images\\mostly_visible_objects"  # Φάκελος εικόνων
mostly_visible_labels_file = "C:\\Machine_Learning_Assignment\\cropped_images\\mostly_visible_labels.txt"  # Labels 
visible_output_csv_path = "C:\\Machine_Learning_Assignment\\output_features\\visible_hog_feature.csv"  # Τελικό αρχείο CSV για visible images


# Δημιουργία HOG descriptor με καθορισμένες παραμέτρους
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
nbins = 9
hog = cv2.HOGDescriptor(_winSize=(128, 64),
                        _blockSize=block_size,
                        _blockStride=block_stride,
                        _cellSize=cell_size,
                        _nbins=nbins)

# Σταθερό μέγεθος εικόνας
target_size = (128, 64)  # (πλάτος, ύψος)

def extract_hog_features(image_path):
    """
    Εξάγει τα HOG features για μία εικόνα.
    """
    # Φορτώνουμε την εικόνα
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Η εικόνα {image_path} δεν βρέθηκε!")
        return None

    # Αλλαγή μεγέθους της εικόνας σε σταθερό μέγεθος
    resized_image = cv2.resize(image, target_size)

    # Εξαγωγή HOG features
    hog_features = hog.compute(resized_image)

    # Μετατροπή σε μονοδιάστατο πίνακα
    return hog_features.flatten()

def load_labels(labels_file):
    """
    Φορτώνει τα labels από ένα αρχείο .txt.
    """
    labels = {}
    with open(labels_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            image_name = parts[0]
            image_label = parts[1]
            labels[image_name] = image_label
    return labels

def process_images(image_folder, labels, csv_writer):
    """
    Επεξεργάζεται εικόνες από έναν φάκελο, εξάγοντας HOG features και αποθηκεύοντας τα στο CSV.
    """
    for image_name, image_label in labels.items():
        image_path = os.path.join(image_folder, image_name)
        features = extract_hog_features(image_path)

        if features is not None:
            # Συνδυάζουμε τα features με την κλάση της εικόνας
            row = features.tolist() + [image_label]
            csv_writer.writerow(row)

# Φορτώνουμε τα labels
visible_labels = load_labels(mostly_visible_labels_file)

# Άνοιγμα των αρχείων CSV για αποθήκευση
with open(visible_output_csv_path, "w", newline="") as visible_csvfile:

    visible_csv_writer = csv.writer(visible_csvfile)

    # Επικεφαλίδες: feature_1, feature_2, ..., feature_N, class
    num_features = len(extract_hog_features(os.path.join(mostly_visible_objects_path, list(visible_labels.keys())[0])))
    header = [f"feature_{i+1}" for i in range(num_features)] + ["class"]
    
    visible_csv_writer.writerow(header)

    # Επεξεργασία εικόνων visible
    print("Επεξεργασία visible images...")
    process_images(mostly_visible_objects_path, visible_labels, visible_csv_writer)

print(f"Η διαδικασία ολοκληρώθηκε! Τα δεδομένα αποθηκεύτηκαν στα {visible_output_csv_path}")

