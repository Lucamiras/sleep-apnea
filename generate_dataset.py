import argparse
import logging
from src.preprocessing.pipeline import DataPreprocessor
from src.preprocessing.steps.config import load_config


logging.basicConfig(level=logging.INFO)

def main(arguments):
    """
    Example usage:
    python generate_dataset.py --configfile "config.yaml"
    """
    config_file = "config.yaml"

    if arguments.configfile:
        config_file = arguments.configfile

    try:
        config = load_config(config_file)
    except Exception as e:
        logging.error("Error during loading of configuration")
        raise e

    pipeline = DataPreprocessor(config)
    pipeline.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f',
        '--configfile',
        action='store_true',
        help='Provide the path of the yaml config file.'
    )
    args = parser.parse_args()
    main(args)