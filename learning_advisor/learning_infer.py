import configparser
import importlib
import json
import logging
import os
import pickle

import numpy as np

from learning_advisor.gym_db.common import EnvironmentType
from learning_advisor.stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from learning_advisor.learning_utils.workload import Query, Workload


BUDGET = 500
VERY_HIGH_BUDGET = 1000000


def pre_infer_obj(exp_load, model_load, env_load, db_conf=None):
    with open(exp_load, "rb") as rf:
        swirl_exp = pickle.load(rf)

    if "max_indexes" not in swirl_exp.exp_config.keys():
        swirl_exp.exp_config["max_indexes"] = 5
    if db_conf is not None:
        swirl_exp.schema.db_config = db_conf

    swirl_model = swirl_exp.model_type.load(model_load)
    swirl_model.training = False

    ParallelEnv = SubprocVecEnv if swirl_exp.exp_config["parallel_environments"] > 1 else DummyVecEnv
    training_env = ParallelEnv([swirl_exp.make_env(env_id,
                                                   environment_type=EnvironmentType.TRAINING,
                                                   workloads_in=None,
                                                   db_config=swirl_exp.schema.db_config)
                                # for env_id in range(1)])
                                for env_id in range(swirl_exp.exp_config["parallel_environments"])])
    swirl_model.set_env(VecNormalize.load(env_load, training_env))
    swirl_model.env.training = False

    return swirl_exp, swirl_model


def get_swirl_res(swirl_exp, query_text, swirl_model):
    if query_text is None:
        eval_workload = swirl_exp.workload_generator.wl_testing[0]
    elif isinstance(query_text, list):
        eval_workload = list()
        for qid, sql in enumerate(query_text):
            query = Query(qid, sql, frequency=1)
            # assign column value to `query` object.
            swirl_exp.workload_generator._store_indexable_columns(query)
            workload = Workload([query], description="")
            workload.budget = BUDGET
            eval_workload.append(workload)
    elif query_text.endswith(".pickle"):
        with open(query_text, "rb") as rf:
            eval_workload = pickle.load(rf)[0]

    n_eval_episodes = len(eval_workload)

    evaluation_env = swirl_exp.DummyVecEnv(
        [swirl_exp.make_env(0, EnvironmentType.TESTING,
                            workloads_in=eval_workload,
                            db_config=swirl_exp.schema.db_config)])
    evaluation_env = swirl_exp.VecNormalize(
        evaluation_env, norm_obs=True, norm_reward=False,
        gamma=swirl_exp.exp_config["rl_algorithm"]["gamma"], training=False
    )

    training_env = swirl_model.get_vec_normalize_env()
    swirl_exp.sync_envs_normalization(training_env, evaluation_env)
    logging.disable(logging.WARNING)
    swirl_exp.evaluate_policy(swirl_model, evaluation_env, n_eval_episodes)
    logging.disable(logging.INFO)

    performances = evaluation_env.get_attr("episode_performances")[0]

    return performances


def get_drlindex_res(drlindex_exp, query_text, drlindex_model):
    eval_workload = list()
    for qid, sql in enumerate(query_text):
        query = Query(qid, sql, frequency=1)
        # assign column value to `query` object.
        drlindex_exp.workload_generator._store_indexable_columns(query)
        workload = Workload([query], description="")
        workload.budget = VERY_HIGH_BUDGET
        eval_workload.append(workload)

    n_eval_episodes = len(eval_workload)

    evaluation_env = drlindex_exp.DummyVecEnv(
        [drlindex_exp.make_env(0, EnvironmentType.TESTING,
                              workloads_in=eval_workload,
                              db_config=drlindex_exp.schema.db_config)])
    evaluation_env = drlindex_exp.VecNormalize(
        evaluation_env, norm_obs=True, norm_reward=False,
        gamma=drlindex_exp.exp_config["rl_algorithm"]["gamma"], training=False
    )

    training_env = drlindex_model.get_vec_normalize_env()
    drlindex_exp.sync_envs_normalization(training_env, evaluation_env)

    logging.disable(logging.WARNING)
    drlindex_exp.evaluate_policy(drlindex_model, evaluation_env, n_eval_episodes)
    logging.disable(logging.INFO)

    performances = evaluation_env.get_attr("episode_performances")[0]

    return performances


def get_dqn_res(dqn_exp, query_text, dqn_model):
    eval_workload = list()
    for qid, sql in enumerate(query_text):
        query = Query(qid, sql, frequency=1)
        # assign column value to `query` object.
        dqn_exp.workload_generator._store_indexable_columns(query)
        workload = Workload([query], description="")
        workload.budget = VERY_HIGH_BUDGET
        eval_workload.append(workload)

    n_eval_episodes = len(eval_workload)

    evaluation_env = dqn_exp.DummyVecEnv(
        [dqn_exp.make_env(0, EnvironmentType.TESTING,
                          workloads_in=eval_workload,
                          db_config=dqn_exp.schema.db_config)])
    evaluation_env = dqn_exp.VecNormalize(
        evaluation_env, norm_obs=True, norm_reward=False,
        gamma=dqn_exp.exp_config["rl_algorithm"]["gamma"], training=False
    )

    training_env = dqn_model.get_vec_normalize_env()
    dqn_exp.sync_envs_normalization(training_env, evaluation_env)

    logging.disable(logging.WARNING)
    dqn_exp.evaluate_policy(dqn_model, evaluation_env, n_eval_episodes)
    logging.disable(logging.INFO)

    performances = evaluation_env.get_attr("episode_performances")[0]

    return performances


def get_eval_env(swirl_exp, swirl_model, qtext_list, budget):
    if qtext_list.endswith(".pickle"):
        with open(qtext_list, "rb") as rf:
            eval_workload = pickle.load(rf)[0]
    else:
        eval_workload = list()
        for qid, sql in enumerate(qtext_list):
            query = Query(qid, sql, frequency=1)
            swirl_exp.workload_generator._store_indexable_columns(query)
            workload = Workload([query], description="")
            workload.budget = budget
            eval_workload.append(workload)

    evaluation_env = swirl_exp.DummyVecEnv(
        [swirl_exp.make_env(0, EnvironmentType.TESTING,
                            workloads_in=eval_workload,
                            db_config=swirl_exp.schema.db_config)])
    evaluation_env = swirl_exp.VecNormalize(
        evaluation_env, norm_obs=True, norm_reward=False,
        gamma=swirl_exp.exp_config["rl_algorithm"]["gamma"], training=False
    )

    training_env = swirl_model.get_vec_normalize_env()
    swirl_exp.sync_envs_normalization(training_env, evaluation_env)

    return evaluation_env
