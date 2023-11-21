from ultralytics import YOLO

import os
import numpy as np
import torch
import gc
import glob

class ModelHandler:
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path

    def train(self, EPOCH, EXP_NAME, IMGSZ=256):
        # train with current model_path, then change the model path to EXP_NAME model.
        model = YOLO(self.model_path)
        model.train(data=self.data_path, epochs=EPOCH, name=EXP_NAME, imgsz=IMGSZ, device=0)

        # Update model_path to point to the newly trained model
        model_path = f"runs/segment/{EXP_NAME}/weights/best.pt"
        self.model_path = os.path.join(os.getcwd(), model_path)

        torch.cuda.empty_cache()
        gc.collect()

    def infer(self, IMG_PATH, OUTPUT_PATH, label_map, batch_size=100):
        # Extract the plant names from the label map
        plants = list(label_map.values())
        # Initialize the model
        model = YOLO(self.model_path)

        for ptype in plants:
            plant_path = os.path.join(IMG_PATH, ptype)

            # Create corresponding directory in OUTPUT_PATH for this plant type
            output_dir = os.path.join(OUTPUT_PATH, ptype)
            os.makedirs(output_dir, exist_ok=True)

            img_files = glob.glob(os.path.join(plant_path, "*.jpg"))  # Assuming images are in .jpg format

            for i in range(0, len(img_files), batch_size): # we're loading and processing them in batches
                batch_files = img_files[i:i + batch_size]
                results = model(batch_files)

                for result, img_file in zip(results, batch_files):
                    if result.masks is not None:
                        masks = result.masks.xyn  # Masks object for segmentation masks outputs
                        cls = result.boxes.cls

                        # Format output and write to file
                        masks_flattened = np.concatenate(masks).flatten()  # concatenate sub-arrays into one long array
                        cls_value = cls.cpu().numpy()[0]  # extract the single value from the cls tensor
                        output_str = f"{cls_value} " + " ".join(map(str, masks_flattened))

                        base_filename = os.path.splitext(os.path.basename(img_file))[0]
                        output_file = os.path.join(output_dir, base_filename + ".txt")

                        with open(output_file, "w") as f:
                            f.write(output_str)
                    else:
                        print(f"No predictions for image: {img_file}")

                torch.cuda.empty_cache()  # Empty CUDA cache after processing each batch
            gc.collect()

    def val(self, TEST_DATA_PATH):
        model = YOLO(self.model_path)
        metrics = model.val(data=TEST_DATA_PATH, device=0)
        torch.cuda.empty_cache()
        gc.collect()
        return metrics
