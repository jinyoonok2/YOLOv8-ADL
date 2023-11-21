from model_handler import ModelHandler
import argparse
import sys
import os

def validate_model(model_path, data_path):
    # Get the model name from the path
    model_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))

    model_handler = ModelHandler(data_path=None, model_path=model_path)
    metrics = model_handler.val(TEST_DATA_PATH=data_path)

    # Create output directory path using metrics.save_dir
    output_dir = os.path.join(metrics.save_dir)

    # Make sure the directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Redirect stdout to a file
    old_stdout = sys.stdout
    sys.stdout = open(os.path.join(output_dir, f'{model_name}_output.txt'), 'w')

    print(metrics)
    print(metrics.box)

    # Restore original stdout
    sys.stdout = old_stdout

    print(f'Metrics and box have been printed to {model_name}_output.txt in the directory: {output_dir}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('MODEL_100', type=str, help="Path to the first model weights")
    parser.add_argument('MODEL_ALL', type=str, help="Path to the second model weights")
    parser.add_argument('TEST_DATA', type=str, help="Path to the test data")
    args = parser.parse_args()

    validate_model(args.MODEL_100, args.TEST_DATA)
    validate_model(args.MODEL_ALL, args.TEST_DATA)



# 1. seg model poly 100 and poly all
# python test_model_100_ALL.py runs\segment\seg-poly-initial\weights\best.pt runs\segment\seg_poly_AL2\weights\best.pt datasets\2_tomato-test\data.yaml

# 2. seg model bb 100 and bb all
# python test_model_100_ALL.py runs\segment\seg-bb-100\weights\best.pt runs\segment\seg-bb-all\weights\best.pt datasets\tomato-obb\data.yaml

# 3. det model poly 100 and poly all
# python test_model_100_ALL.py runs\detect\det-poly-initial\weights\best.pt runs\detect\det-poly-AL2\weights\best.pt datasets\2_tomato-test\data.yaml

# 4. det model bb 100 and bb all
# python test_model_100_ALL.py runs\detect\det-bb-100\weights\best.pt runs\detect\det-bb-all\weights\best.pt datasets\tomato-obb\data.yaml


