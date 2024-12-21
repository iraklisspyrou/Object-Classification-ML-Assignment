# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 10:27:33 2024

@author: irakl
"""

import os
import cv2

# Ορισμός διαδρομών
images_path = "C:\\Machine_Learning_Assignment\\image_2"  # Φάκελος εικόνων
labels_path = "C:\\Machine_Learning_Assignment\\label_2"  # Φάκελος labels
output_path = "C:\\Machine_Learning_Assignment\\cropped_images"  # Φάκελος εξόδου

# Δημιουργία φάκελου εξόδου
os.makedirs(os.path.join(output_path, "mostly_visible_objects"), exist_ok=True)

# Δημιουργία αρχείου για τα labels
mostly_visible_labels_path = os.path.join(output_path, "mostly_visible_labels.txt")


with open(mostly_visible_labels_path, "w") as visible_file:
    # Επεξεργασία αρχείων
    for label_file in os.listdir(labels_path):
        if not label_file.endswith(".txt"):
            continue

        # Διαβάζουμε τα labels
        label_path = os.path.join(labels_path, label_file)
        with open(label_path, "r") as f:
            lines = f.readlines()

        # Φορτώνουμε την αντίστοιχη εικόνα
        image_file = label_file.replace(".txt", ".png")
        image_path = os.path.join(images_path, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Η εικόνα {image_file} δεν βρέθηκε!")
            continue

        # Επεξεργασία γραμμών
        for i, line in enumerate(lines):
            parts = line.strip().split()
            object_class = parts[0]
            truncation = float(parts[1])  #πόσο κομμένο είναι το object στην εικόνα
            occlusion=int(parts[2]) #πόσο κρυμμένο είναι το object από άλλα objects
            bbox = list(map(float, parts[4:8]))  # Συντεταγμένες bounding box

            # Αγνοούμε την κλάση DontCare
            if object_class == "DontCare" or object_class=="Misc" or object_class=="Van" or object_class=="Truck" or object_class=="Person_sitting":
                continue
            if occlusion>1 or truncation>0.40:
                continue
            # Συντεταγμένες bounding box
            x_min, y_min, x_max, y_max = map(int, bbox)
            cropped = image[y_min:y_max, x_min:x_max]
            
            #Εγγραφή στο αρχείο εξόδου
            folder = "mostly_visible_objects"
            label_file_to_write = visible_file
          

            # Αποθήκευση cropped εικόνας
            cropped_filename = f"{os.path.splitext(image_file)[0]}_{i}.png"
            cropped_path = os.path.join(output_path, folder, cropped_filename)
            cv2.imwrite(cropped_path, cropped)

            # Προσθήκη του label στο αντίστοιχο αρχείο
            label_line = f"{cropped_filename} {object_class}\n"
            label_file_to_write.write(label_line)

print("Η διαδικασία ολοκληρώθηκε!")