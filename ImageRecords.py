import os
import tensorflow as tf
from PIL import Image
import numpy as np
import BatchDataset as BD

def get_records_NYU_RGB(data_dir, type):
    """
    get records according to NYU dataset (image and gt)
    :param data_dir:
    :return:
    """
    result_records = []
    f = open(os.path.join(data_dir, "image-"+type+".lst"))
    for line in f.readlines():
        if type == "train":
            record = {}
            record["image"] = os.path.join(data_dir, line.split()[0])
            record["annotation"] = os.path.join(data_dir, line.split()[1])
            record["filename"] = line.split()[0][-12:]
            result_records.append(record)
        elif type == "test":
            record = {}
            record["image"] = os.path.join(data_dir, line.strip())
            record["filename"] = line.strip()[-12:]
            record["annotation"] = ""
            result_records.append(record)
    return result_records

def get_records_NYU_HHA(data_dir, type):
    """
    get records according to NYU dataset (image and gt)
    :param data_dir:
    :return:
    """
    result_records = []
    f = open(os.path.join(data_dir, "hha-"+type+".lst"))
    for line in f.readlines():
        if type == "train":
            record = {}
            record["image"] = os.path.join(data_dir, line.split()[0])
            record["annotation"] = os.path.join(data_dir, line.split()[1])
            record["filename"] = line.split()[0][-12:]
            result_records.append(record)
        elif type == "test":
            record = {}
            record["image"] = os.path.join(data_dir, line.strip())
            record["filename"] = line.strip()[-12:]
            record["annotation"] = ""
            result_records.append(record)
    return result_records

def get_records_BSDS_train(data_dir):
    result_records = []
    f = open(os.path.join(data_dir, "train_pair.lst"))
    for line in f.readlines():
        record = {}
        record["image"] = os.path.join(data_dir, line.split()[0])
        record["annotation"] = os.path.join(data_dir, line.split()[1])
        record["filename"] = line.split()[0].split('/')[-1]
        result_records.append(record)
    return result_records

def get_records_BSDS_test(data_dir):
    result_records = []
    f = open(os.path.join(data_dir, "test.lst"))
    for line in f.readlines():
        record = {}
        record["image"] = os.path.join(data_dir, line.split()[0])
        record["annotation"] = ""
        record["filename"] = line.split()[0].split('/')[-1]
        result_records.append(record)
    return result_records


if __name__ == '__main__':
    a = get_records_NYU_RGB("/home/zhouzhilong/NYUD", "test")
    print(a)


