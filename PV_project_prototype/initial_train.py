from model_handler import ModelHandler
import os
import argparse

if __name__ == '__main__':
    # parsing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('DATA_PATH', type=str, help="initial data path")
    parser.add_argument('MODEL_PATH', type=str, help="initial model path")
    parser.add_argument('EXP_NAME', type=str, help="experiment name")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = args.MODEL_PATH
    DATA_PATH = args.DATA_PATH
    EXP_NAME = args.EXP_NAME

    EPOCH = 10
    IMGSZ = 256

    # Path to the trained model
    trained_model_path = f"runs/segment/{EXP_NAME}/weights/best.pt"

    # initialize ModelHandler with the initial model and data paths
    model_handler = ModelHandler(DATA_PATH, MODEL_PATH)

    model_handler.train(EPOCH, EXP_NAME, IMGSZ)


# 1.
# python initial_train.py datasets\0_tomato-train-100\data.yaml yolov8s-seg.pt seg-poly-initial
# continue in active_learning.py

# 2.
# python initial_train.py datasets\0_tomato-train-100\data.yaml yolov8s.pt det-poly-initial
# python initial_train.py outputs\initial_output\data.yaml runs\detect\det-poly-initial\weights\best.pt det-poly-AL1
# python initial_train.py outputs\AL_output_1\data.yaml runs\detect\det-poly-AL1\weights\best.pt det-poly-AL2

# 3.
# python initial_train.py datasets\tomato-obb-100\data.yaml yolov8s-seg.pt seg-bb-100
# python initial_train.py datasets\tomato-obb\data.yaml yolov8s-seg.pt seg-bb-all
# python initial_train.py datasets\tomato-obb-100\data.yaml yolov8s.pt det-bb-100
# python initial_train.py datasets\tomato-obb\data.yaml yolov8s.pt det-bb-all

