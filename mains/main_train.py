
import __init__
import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.example_model import ExampleModel
from trainers.example_trainer import ExampleTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args

import configparser


def main():
    cp = configparser.ConfigParser()
    cp.read("test.ini")

    secs = cp.sections()
    print(cp.sections())

    for sec in secs:
        opts = cp.options(sec)
        for opt in opts:
            val = cp.get(sec, opt)
            val += "test....."
            cp.set(sec, opt, val)
            print(sec, opt)

    cp.write(open("out.ini", "w"))


if __name__ == '__main__':
    main()
