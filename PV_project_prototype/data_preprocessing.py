import os
from glob import glob
import glob
import shutil
import yaml

def correct_data(LABEL_PATH, label_map):
    # Check if the path exists and is a directory
    if not os.path.exists(LABEL_PATH) or not os.path.isdir(LABEL_PATH):
        print(f"No such directory: {LABEL_PATH}")
        return

    reverse_label_map = {v: k for k, v in label_map.items()}  # Reverse mapping

    # 1. detect the error, then correct them into correct label class
    for disease in label_map.values():
        dir_path = os.path.join(LABEL_PATH, disease)

        # Get all text files in the directory
        text_files = glob.glob(os.path.join(dir_path, f"{disease}_*.txt"))
        num_errors = 0

        # If there are no labels in the directory, print a notification and skip to the next disease
        if not text_files:
            print(f"No labels found in directory: {dir_path}")
            continue

        total_labels = len(text_files)

        # Iterate over each text file
        for file_path in text_files:
            with open(file_path, "r") as f:
                contents = f.read()
                first_char = contents[0]

                # Check if the first character matches the corresponding value in reverse_label_map
                if int(first_char) != reverse_label_map[disease]:
                    num_errors += 1

                    # Replace the first character with the corresponding value from the dictionary
                    contents = str(reverse_label_map[disease]) + contents[1:]
                    with open(file_path, "w") as f2:
                        f2.write(contents)

        error_percentage = ((total_labels - num_errors) / total_labels) * 100
        print(f"{disease}: {num_errors} errors out of {total_labels}: ({error_percentage:.2f}% of labels success)")


def split_data(LABEL_PATH, IMG_PATH, label_map, split_ratio=0.8):

    diseases = label_map.values()
    UNANNOTATED_IMAGES = os.path.join(LABEL_PATH, "unannotated")

    # Create subdirectories for images and labels
    TRAIN_IMAGES = os.path.join(LABEL_PATH, "train/images")
    TRAIN_LABELS = os.path.join(LABEL_PATH, "train/labels")
    VALID_IMAGES = os.path.join(LABEL_PATH, "valid/images")
    VALID_LABELS = os.path.join(LABEL_PATH, "valid/labels")
    os.makedirs(TRAIN_IMAGES, exist_ok=True)
    os.makedirs(TRAIN_LABELS, exist_ok=True)
    os.makedirs(VALID_IMAGES, exist_ok=True)
    os.makedirs(VALID_LABELS, exist_ok=True)

    for disease in diseases:
        img_dir_path = os.path.join(IMG_PATH, disease)
        labels_dir_path = os.path.join(LABEL_PATH, disease)

        # Create disease directories under unannotated
        disease_unannotated = os.path.join(UNANNOTATED_IMAGES, disease)
        os.makedirs(disease_unannotated, exist_ok=True)

        # Check if the label directory is empty, if so skip the current disease
        if not os.listdir(labels_dir_path):
            print(f"no detected labels found in {labels_dir_path}")
            image_files = glob.glob(os.path.join(img_dir_path, "*.jpg"))
            for image_file in image_files:
                dst_file_path = os.path.join(disease_unannotated, os.path.basename(image_file))
                if not os.path.exists(dst_file_path):  # Check if the file doesn't already exist
                    shutil.copy(image_file, dst_file_path)
            continue

        # Get all image files and their corresponding label files in the directory
        image_files = glob.glob(os.path.join(img_dir_path, "*.jpg"))
        label_files = glob.glob(os.path.join(labels_dir_path, "*.txt"))

        # Collect all base names (without extension) for label files
        label_names = [os.path.splitext(os.path.basename(label_file))[0] for label_file in label_files]

        num_labels_to_move = int(split_ratio * len(label_files))

        # Split the label files into training and validation sets
        train_labels = label_files[:num_labels_to_move]
        valid_labels = label_files[num_labels_to_move:]

        # Copy label files to corresponding directories
        for label_file in train_labels:
            dst_file_path = os.path.join(TRAIN_LABELS, os.path.basename(label_file))
            if not os.path.exists(dst_file_path):  # Check if the file doesn't already exist
                shutil.copy(label_file, dst_file_path)

        for label_file in valid_labels:
            dst_file_path = os.path.join(VALID_LABELS, os.path.basename(label_file))
            if not os.path.exists(dst_file_path):  # Check if the file doesn't already exist
                shutil.copy(label_file, dst_file_path)

        # Check each image file
        for image_file in image_files:
            image_name = os.path.splitext(os.path.basename(image_file))[0]
            # Check if there is a corresponding label file
            if image_name in label_names:
                if image_name + '.txt' in [os.path.basename(label) for label in train_labels]:
                    dst_file_path = os.path.join(TRAIN_IMAGES, os.path.basename(image_file))
                    if not os.path.exists(dst_file_path):  # Check if the file doesn't already exist
                        shutil.copy(image_file, dst_file_path)

                elif image_name + '.txt' in [os.path.basename(label) for label in valid_labels]:
                    dst_file_path = os.path.join(VALID_IMAGES, os.path.basename(image_file))
                    if not os.path.exists(dst_file_path):  # Check if the file doesn't already exist
                        shutil.copy(image_file, dst_file_path)
            else:
                # If no corresponding label exists, move the image to the unannotated disease directory
                dst_file_path = os.path.join(disease_unannotated, os.path.basename(image_file))
                if not os.path.exists(dst_file_path):  # Check if the file doesn't already exist
                    shutil.copy(image_file, dst_file_path)


def generate_data_yaml(train_dir, valid_dir, label_map, save_path):
    class_names = list(label_map.values())
    data = {
        'train': train_dir,
        'val': valid_dir,
        'nc': len(class_names),
        'names': class_names
    }

    with open(save_path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
