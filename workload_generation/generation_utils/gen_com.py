import configparser
import copy

import torch
import json
import argparse

import random
import logging
import tensorflow as tf

from workload_generation.generation_utils import data_loader

tf_step = 0
summary_writer = None

def get_conf(conf_file):
    conf = configparser.ConfigParser()
    conf.read(conf_file)

    return conf


def get_parser():
    parser = argparse.ArgumentParser(
        description="The Framework of Adversarial Workload Generation.")

    parser.add_argument("--gpu_no", type=str, default="-1")
    parser.add_argument("--exp_id", type=str, default="adv_exp_id")
    parser.add_argument("--train_mode", type=str, default="rl_pg",
                        choices=["pre_train", "rl_pg"])
    parser.add_argument("--model_struct", type=str, default="Seq2Seq",
                        choices=["Seq2Seq", "SingleRNN"])
    parser.add_argument("--is_bid", action="store_true")
    parser.add_argument("--is_attn", action="store_true")
    parser.add_argument("--is_ptr", action="store_true")

    parser.add_argument("--rnn_type", type=str, default="GRU")

    parser.add_argument("--pre_epoch", type=int, default=1)
    parser.add_argument("--pre_lr", type=float, default=0.001)
    parser.add_argument("--pre_mode", type=str, default="not_ae")
    parser.add_argument("--force_ratio", type=float, default=0.7)

    parser.add_argument("--rein_epoch", type=int, default=1)
    parser.add_argument("--rein_lr", type=float, default=0.001)

    parser.add_argument("--max_diff", type=int, default=5)
    parser.add_argument("--pert_mode", type=str, default="column",
                        choices=["all", "value", "column"])
    parser.add_argument("--reward", type=str, default="dynamic",
                        choices=["static", "dynamic", "all_dynamic"])
    parser.add_argument("--reward_form", type=str, default="cost_red_ratio",
                        choices=["cost_red_ratio", "cost_ratio", "cost_ratio_norm", "inv_cost_ratio_norm"])

    parser.add_argument("--work_level", type=str, default="query")
    parser.add_argument("--work_type", type=str, default="not_template")
    parser.add_argument("--inf", type=int, default=1e6)
    parser.add_argument("--eps", type=int, default=1e-36)

    parser.add_argument("--db_file", type=str,
                        default="./data_resource/database_conf/db_info.conf")
    parser.add_argument("--data_load", type=str,
                        default="./data_resource/sample_data/sample_data.pt")

    parser.add_argument("--model_load", type=str, default="empty")

    parser.add_argument("--cost_estimator", type=str, default="optimizer")
    parser.add_argument("--reward_base", action="store_true")

    # params for victim.
    # i) heuristic-based method: `extend`.
    parser.add_argument("--victim", type=str, default="extend")
    parser.add_argument("--exp_file", type=str,
                        default="./data_resource/heuristic_conf/extend_config.json")
    parser.add_argument("--sel_param", type=str, default="parameters")

    # ii) learning-based method: `swirl`.
    parser.add_argument("--baseline", type=str, default="extend")
    parser.add_argument("--base_exp_file", type=str,
                        default="./data_resource/heuristic_conf/extend_config.json")
    parser.add_argument("--swirl_exp_load", type=str,
                        default="")
    parser.add_argument("--swirl_model_load", type=str,
                        default="")
    parser.add_argument("--swirl_env_load", type=str,
                        default="")

    parser.add_argument("--vocab_mode", type=str, default="without_table",
                        choices=["with_table", "without_table"])
    parser.add_argument("--colinfo_file", type=str,
                        default="./data_resource/database_conf/colinfo.json")
    parser.add_argument("--wordinfo_file", type=str,
                        default="./data_resource/vocab/wordinfo.json")
    parser.add_argument("--schema_file", type=str,
                        default="./data_resource/database_conf/schema.json")
    parser.add_argument("--data_save", type=str,
                        default="./workload_generation/exp_res/{}/data/{}_data.pt")

    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--runlog", type=str,
                        default="./workload_generation/exp_res/{}/exp_runtime.log")
    parser.add_argument("--logdir", type=str,
                        default="./workload_generation/exp_res/{}/logdir/")
    parser.add_argument("--model_save_gap", type=int, default=1)
    parser.add_argument("--model_save", type=str,
                        default="./workload_generation/exp_res/{}/model/rewrite_{}.pt")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--max_len", type=int, default=55)
    parser.add_argument("--src_vbs", type=int, default=3040)
    parser.add_argument("--tgt_vbs", type=int, default=3040)
    parser.add_argument("--emb_size", type=int, default=128)

    parser.add_argument("--act_hsz", type=int, default=128)
    parser.add_argument("--act_nls", type=int, default=1)
    parser.add_argument("--act_bid", type=bool, default=False)

    parser.add_argument("--enc_hidden_size", type=int, default=128)
    parser.add_argument("--dec_hidden_size", type=int, default=128)

    parser.add_argument("--cri_hsz", type=int, default=128)
    parser.add_argument("--cri_nls", type=int, default=1)
    parser.add_argument("--grad_norm", type=int, default=5.)
    parser.add_argument("--GAMMA", type=float, default=0.90)
    parser.add_argument("--LAMBDA", type=float, default=0.01)
    parser.add_argument("--TAU", type=float, default=1.0)

    return parser


def set_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # log to file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)


def add_summary_value(key, value, step=None):
    if step is None:
        summary_writer.add_scalar(key, value, tf_step)
    else:
        summary_writer.add_scalar(key, value, step)


def find_parameter_list(algorithm_config):
    parameters = algorithm_config["parameters"]
    configs = []
    if parameters:
        # if more than one list --> raise
        # Only support one param list in each algorithm.
        counter = 0
        for key, value in parameters.items():
            if isinstance(value, list):
                counter += 1
        if counter > 1:
            raise Exception("Too many parameter lists in config.")

        for key, value in parameters.items():
            if isinstance(value, list):
                for i in value:
                    new_config = copy.deepcopy(algorithm_config)
                    new_config["parameters"][key] = i
                    configs.append(new_config)
    if len(configs) == 0:
        configs.append(algorithm_config)

    return configs
