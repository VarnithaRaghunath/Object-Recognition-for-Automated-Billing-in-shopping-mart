import csv
import os
from PIL import Image

# Path to your dataset
dataset_path = r'C:\Users\Vibha Raghunath\OneDrive\Desktop\Varnitha\project\dataset'
annotations_path = 'annotations.csv'

# Open the CSV file for writing
with open(annotations_path, 'w', newline='') as csvfile:
    fieldnames = ['filename', 'width', 'height', 'label', 'xmin', 'ymin', 'xmax', 'ymax']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Walk through all subdirectories and files in the dataset path
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image file types
                image_path = os.path.join(root, file)
                
                # Open the image to get its dimensions
                with Image.open(image_path) as img:
                    width, height = img.size

                # Assuming label is the folder name containing the image
                label = os.path.basename(root)

                # Example bounding box coordinates for demonstration (Full image bounding box)
                xmin, ymin, xmax, ymax = 0, 0, width, height

                # Write the annotation to the CSV file
                writer.writerow({
                    'filename': file,
                    'width': width,
                    'height': height,
                    'label': label,
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax
                })

print("Annotations CSV file created successfully!")
