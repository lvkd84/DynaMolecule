import torch
import configparser
import argparse

def main(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)

    model_path = config.get('PATH','ModelFilePath')
    data_path = config.get('PATH','ProcessedEvalDataFilePath')
    saving_path = config.get('PATH','ResultPath')
    labeled = config.get('OPTIONS','Labeled')

    predictor = torch.load(model_path)
    predictor.evaluate(data_path, save_result_path = saving_path, labeled = labeled, signal_obj = None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True)
    args = parser.parse_args()

    main(args.config_path)