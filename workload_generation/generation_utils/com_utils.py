import configparser
import json
import pickle
import re

import pandas as pd


def multi_load(file_path):
    if file_path.endswith(".json"):
        with open(file_path, "r") as f:
            data = json.load(f)

    elif file_path.endswith(".pickle"):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

    elif file_path.endswith(".csv"):
        data = pd.read_csv(file_path, sep=";")

    else:
        data = open(file_path, "r").readlines()

    return data


def alter_csv(from_path, dst_path):
    data = pd.read_csv(from_path, sep=";")
    data.to_csv(dst_path, sep=",", index=False)


def get_conf(conf_file):
    conf = configparser.ConfigParser()
    conf.read(conf_file)

    return conf


def re_extract():
    pat = re.compile(r"r(.*)\.sql")
    s = "r10.sql"

    return pat.findall(s)
