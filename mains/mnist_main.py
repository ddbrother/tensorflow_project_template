
import __init__
import tensorflow as tf

from data_loader.mnist_data_generator import DataGenerator
from models.mnist_model import ExampleModel
from trainers.mnist_trainer import ExampleTrainer

from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args

import os

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    if os.path.exists(config.summary_dir):
        os.removedirs(config.summary_dir)
    if os.path.exists(config.checkpoint_dir):
        os.removedirs(config.checkpoint_dir)
    create_dirs([config.summary_dir, config.checkpoint_dir])
    
    # create tensorflow session
    sess = tf.Session()
    
    # create your data generator
    data = DataGenerator(config)
    
    # create an instance of the model you wantVGGModel
    model = ExampleModel(config)
    
    # create tensorboard logger
    logger = Logger(sess, config)
    
    # create trainer and pass all the previous components to it
    trainer = ExampleTrainer(sess, model, data, config, logger)
    
    #load model if exists
    model.load(sess)
    
    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
