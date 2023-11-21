import os
import glob
from model_handler import ModelHandler
from data_preprocessing import correct_data, split_data, generate_data_yaml
from ultralytics import YOLO
import argparse


class ActiveLearning:
    def __init__(self, initial_predict_path, abs_path):
        self.unannotated_path = initial_predict_path
        self.abs_path = abs_path
        self.label_path = os.path.join(self.abs_path, "outputs\initial_output")
        self.model_handler = ModelHandler(None, os.path.join("runs", "segment", "seg-poly-initial", "weights", "best.pt"))
        self.label_map = YOLO(self.model_handler.model_path).model.names  # Create label map using YOLO model

    def process(self, EPOCH, IMGSZ):
        # initial variables
        run_counter = 1
        prev_remaining_images = -1

        while True:
            # Perform prediction on the unannotated folder
            self.model_handler.infer(IMG_PATH=self.unannotated_path, OUTPUT_PATH=self.label_path,
                                     label_map=self.label_map)

            # Correct the data
            correct_data(LABEL_PATH=self.label_path, label_map=self.label_map)

            # Split the data into training and validation sets
            split_data(LABEL_PATH=self.label_path, IMG_PATH=self.unannotated_path, label_map=self.label_map)

            # Generate the data.yaml file
            data_yaml_path = os.path.join(self.label_path, "data.yaml")
            generate_data_yaml(train_dir=os.path.join(self.label_path, "train"),
                               valid_dir=os.path.join(self.label_path, "valid"),
                               label_map=self.label_map,
                               save_path=data_yaml_path)

            # Update data_path for the current unannotated data
            self.model_handler.data_path = data_yaml_path
            print(self.model_handler.data_path)

            # Update the list of all disease subdirectories in the unannotated folder
            self.unannotated_path = os.path.join(self.label_path, "unannotated")
            disease_subdirs = [os.path.join(self.unannotated_path, disease) for disease in os.listdir(self.unannotated_path)]

            # Check if there are any images left in the subdirectories
            remaining_images = sum([len(glob.glob(os.path.join(subdir, "*.jpg"))) for subdir in disease_subdirs])

            if remaining_images == 0 or remaining_images == prev_remaining_images:
                break  # Exit the loop if no images are left or if the number of images hasn't changed

            # update the count of prev remaining images
            prev_remaining_images = remaining_images

            # Perform training on current unannotated data
            EXP_NAME = f"seg_poly_AL{run_counter}"
            self.model_handler.train(EPOCH=EPOCH, EXP_NAME=EXP_NAME, IMGSZ=IMGSZ)

            # Update model_path for the current AL model
            self.model_handler.model_path = os.path.join("runs", "segment", EXP_NAME, "weights", "best.pt")

            # Update label path for the next loop
            LABEL_PATH = f"outputs\AL_output_{run_counter}"
            self.label_path = os.path.join(self.abs_path, LABEL_PATH)

            run_counter += 1

        print("Active learning process completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('DATA_PATH', type=str, help="initial data path")
    args = parser.parse_args()

    cwd_path = os.getcwd()
    print(cwd_path)
    active_learning = ActiveLearning(initial_predict_path=args.DATA_PATH, abs_path=cwd_path)
    active_learning.process(EPOCH=10, IMGSZ=256)

# python active_learning.py datasets\1_tomato-active-learning

