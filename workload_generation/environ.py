import torch
import copy
import json
import numpy as np

import pickle
import traceback
import logging

from workload_generation.generation_utils import constants
from workload_generation.generation_utils.mod_sql import vec2sql

from learning_advisor.gym_db.common import EnvironmentType
from learning_advisor.stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from learning_advisor.learning_infer import get_swirl_res
from learning_advisor.learning_infer import get_drlindex_res
from learning_advisor.learning_infer import get_dqn_res

from heuristic_advisor.heuristic_utils import selec_com
from heuristic_advisor.heuristic_utils.postgres_dbms import PostgresDatabaseConnector
from heuristic_advisor.heuristic_utils.workload import Workload

from heuristic_advisor.heuristic_algos.auto_admin_algorithm import AutoAdminAlgorithm
from heuristic_advisor.heuristic_algos.db2advis_algorithm import DB2AdvisAlgorithm
from heuristic_advisor.heuristic_algos.drop_heuristic_algorithm import DropHeuristicAlgorithm
from heuristic_advisor.heuristic_algos.extend_algorithm import ExtendAlgorithm
from heuristic_advisor.heuristic_algos.relaxation_algorithm import RelaxationAlgorithm
from heuristic_advisor.heuristic_algos.anytime_algorithm import AnytimeAlgorithm


ALGORITHMS = {
    "auto_admin": AutoAdminAlgorithm,
    "db2advis": DB2AdvisAlgorithm,
    "drop": DropHeuristicAlgorithm,
    "extend": ExtendAlgorithm,
    "relaxation": RelaxationAlgorithm,
    "anytime": AnytimeAlgorithm
}


class DBEnviron:
    def __init__(self, args):
        self.args = args

        self.exp_conf = None
        self.base_exp_conf = None  # for baseline method
        self.connector = None
        self.columns = None

        # for rl-based victims like `swirl`.
        self.swirl_exp = None
        self.swirl_model = None

        # for learned cost estimator.
        self.cost_model = None

        logging.info(f"The workload level is `{self.args.work_level}({self.args.work_type})`.")
        logging.info(f"The index selection victim algorithm is `{self.args.victim}`.")
        logging.info(f"The mode of evaluated reward is `{self.args.reward}"
                     f"-({self.args.reward_form}, base:{self.args.reward_base})`.")
        logging.info(f"The max difference between `src` and `tgt` is `{self.args.max_diff}`.")
        logging.info(f"The perturbation mode between `src` and `tgt` is `{self.args.pert_mode}`.")

    def setup(self, autocommit=True):
        # 1) the heuristic victim.
        if self.args.victim in constants.heuristic:
            with open(self.args.exp_file, "r") as rf:
                self.exp_conf = json.load(rf)
            logging.disable(logging.DEBUG)
            logging.info(f"Load the exp_conf of heuristic victim `{self.args.victim}` "
                         f"from `{self.args.exp_file}`.")
            logging.info(f"The parameters' key of the heuristic algorithms is `{self.args.sel_param}`.")
            logging.disable(logging.INFO)

            if self.args.reward_base:
                # load the `heuristic baseline` configuration.
                with open(self.args.base_exp_file, "r") as rf:
                    self.base_exp_conf = json.load(rf)
                logging.disable(logging.DEBUG)
                logging.info(f"Load the base_exp_conf of heuristic baseline `{self.args.baseline}` "
                             f"from `{self.args.base_exp_file}`.")
                logging.disable(logging.INFO)

        # 2) the RL-based victim: mcts.
        elif "mcts" in self.args.victim:
            raise NotImplementedError

            # load the `heuristic baseline` configuration.
            with open(self.args.base_exp_file, "r") as rf:
                self.base_exp_conf = json.load(rf)
            logging.disable(logging.DEBUG)
            logging.info(f"Load the exp_conf of heuristic baseline `{self.args.baseline}` "
                         f"from `{self.args.base_exp_file}`.")
            logging.disable(logging.INFO)

            # load the `rl-based victim` configuration.
            logging.disable(logging.DEBUG)
            logging.info(f"Load the exp_conf of RL-based victim `{self.args.victim}` "
                         f"({self.args.budget}, {self.args.select_policy}, {self.args.best_policy}).")
            logging.disable(logging.INFO)

        # 3) the RL-based victim: swirl, drlindex, dqn.
        elif self.args.victim in constants.rl_based:
            # load the `heuristic baseline` configuration.
            with open(self.args.base_exp_file, "r") as rf:
                self.base_exp_conf = json.load(rf)
            logging.disable(logging.DEBUG)
            logging.info(f"Load the exp_conf of heuristic baseline `{self.args.baseline}` "
                         f"from `{self.args.base_exp_file}`.")
            logging.disable(logging.INFO)

            # load the `rl-based victim` configuration.
            with open(self.args.swirl_exp_load, "rb") as rf:
                self.swirl_exp = pickle.load(rf)
            logging.disable(logging.DEBUG)
            logging.info(f"Load the exp_conf of RL-based victim `{self.args.victim}` "
                         f"from `{self.args.swirl_exp_load}`.")
            logging.disable(logging.INFO)

            self.swirl_model = self.swirl_exp.model_type.load(self.args.swirl_model_load)
            self.swirl_model.training = False
            logging.disable(logging.DEBUG)
            logging.info(f"Load the pretrained model from `{self.args.swirl_model_load}`.")
            logging.disable(logging.INFO)

            if "max_indexes" not in self.swirl_exp.exp_config.keys():
                self.swirl_exp.exp_config["max_indexes"] = 5

            ParallelEnv = SubprocVecEnv if self.swirl_exp.exp_config["parallel_environments"] > 1 else DummyVecEnv
            training_env = ParallelEnv([self.swirl_exp.make_env(env_id,
                                                                environment_type=EnvironmentType.TRAINING,
                                                                workloads_in=None,
                                                                # db_config=selec_com.get_conf(self.args.db_file))
                                                                db_config=self.swirl_exp.schema.db_config)
                                        for env_id in range(self.swirl_exp.exp_config["parallel_environments"])])
            self.swirl_model.set_env(VecNormalize.load(self.args.swirl_env_load, training_env))
            self.swirl_model.env.training = False
            logging.disable(logging.DEBUG)
            logging.info(f"Load the normalized env from {self.args.swirl_env_load}.")
            logging.disable(logging.INFO)

        db_conf = selec_com.get_conf(self.args.db_file)
        logging.disable(logging.DEBUG)
        logging.info(f"Load the db_conf from `{self.args.db_file}`.")
        logging.disable(logging.INFO)

        self.connector = PostgresDatabaseConnector(db_conf, autocommit=autocommit)

        logging.disable(logging.DEBUG)
        self.connector.drop_indexes()
        logging.disable(logging.INFO)

        tables, self.columns = selec_com.get_columns_from_schema(self.args.schema_file)
        logging.disable(logging.DEBUG)
        logging.info(f"Load the schema from `{self.args.schema_file}`.")
        logging.disable(logging.INFO)

        logging.disable(logging.DEBUG)
        logging.info(f"The mode of the cost estimator is `{self.args.cost_estimator}`.")
        logging.disable(logging.INFO)

    def step(self, decoded_words, sql_tokens, last_reward, idx2word, col_info):
        rewards = list()
        for qi in range(len(sql_tokens)):
            tgt_vec = copy.deepcopy(sql_tokens[qi]["pno_tokens"])
            tgt_vec[:len(decoded_words[qi])] = decoded_words[qi]

            if sql_tokens[qi]["pno_tokens"] == tgt_vec:
                rewards.append(0.)
            elif len(sql_tokens[qi]["pno_tokens"]) < len(decoded_words[qi]):
                rewards.append(last_reward[qi])
            elif sql_tokens[qi]["pno_tokens"][len(decoded_words[qi]) - 1] == decoded_words[qi][-1]:
                rewards.append(last_reward[qi])
            else:
                try:
                    tgt_sql = vec2sql([sql_tokens[qi]], [tgt_vec], idx2word, col_info)[0]["sql_text"]
                    self.connector.get_ind_cost(tgt_sql, "")
                    rewards.append(self.get_index_reward(sql_tokens[qi], tgt_vec, idx2word, col_info))
                except:
                    rewards.append(0.)

        return rewards

    def get_cost_feat(self, plan_info, base_plan_info):
        feat_chan = list()
        if "cost" in self.args.cost_feat_type:
            feat_chan.append("node_cost_sum")
            feat_chan.append("node_row_sum")
            feat_chan.append("node_cost_wsum")
            feat_chan.append("node_row_wsum")
        elif "cost" in self.args.cost_feat_type:
            feat_chan.append("node_cost_sum")
            feat_chan.append("node_cost_wsum")
        elif "row" in self.args.cost_feat_type:
            feat_chan.append("node_row_sum")
            feat_chan.append("node_row_wsum")

        # Cost and Row
        plan_feat, base_plan_feat = list(), list()
        for chan in feat_chan:
            plan_node_feat = np.zeros((len(constants.ops),))
            for node in plan_info[chan].keys():
                if node in constants.ops:
                    plan_node_feat[constants.ops.index(node)] = plan_info[chan][node]
                else:
                    logging.error(f"New ops`({node})` occurred!")
            plan_feat.extend(plan_node_feat)

            base_plan_node_feat = np.zeros((len(constants.ops),))
            for node in base_plan_info[chan].keys():
                if node in constants.ops:
                    base_plan_node_feat[constants.ops.index(node)] = base_plan_info[chan][node]
                else:
                    logging.error(f"New ops`({node})` occurred!")
            base_plan_feat.extend(base_plan_node_feat)

        if "concat" in self.args.cost_feat_type:
            plan_feat.extend(base_plan_feat)
        elif "diff" in self.args.cost_feat_type:
            plan_feat = list(np.array(plan_feat) - np.array(base_plan_feat))

        scaler = torch.load(self.args.cost_scale_load)
        plan_feat = np.array(scaler.transform([plan_feat]), dtype=np.float32)

        return plan_feat

    def get_heur_indexes(self, algo, exp_conf, work_list):
        workload = Workload(selec_com.read_row_query(work_list, exp_conf,
                                                     self.columns, type=self.args.work_type))
        config = selec_com.find_parameter_list(exp_conf["algorithms"][0], params=self.args.sel_param)[0]
        victim = ALGORITHMS[algo](self.connector, config["parameters"])

        # format the index returned in the `tbl#col1,col2` form.
        indexes = [str(ind) for ind in victim.calculate_best_indexes(workload)]
        cols = [ind.split(",") for ind in indexes]
        cols = [list(map(lambda x: x.split(".")[-1], col)) for col in cols]
        indexes = [f"{ind.split('.')[0]}#{','.join(col)}" for ind, col in zip(indexes, cols)]

        return indexes

    def get_heur_result(self, sql_token, tgt_sql):
        cost_ratio = list()
        no_cost, ind_cost = list(), list()
        # 1) basic: comparison against the `without index` case.
        if not self.args.reward_base:
            # get the index recommendation result (src_workload) before perturbation from the scratch.
            if self.args.reward == "all_dynamic":
                if self.args.work_level == "query":
                    work_list = [sql_token["sql"]]
                elif self.args.work_level == "workload":
                    work_list = [sql_token["sql"]] + sql_token["workload"]["sql"]

                indexes = self.get_heur_indexes(self.args.victim, self.exp_conf, work_list)

                if self.args.cost_estimator == "optimizer":
                    no_cost_, ind_cost_ = 0, 0
                    for sql in work_list:
                        no_cost_ += self.connector.get_ind_cost(sql, "")
                        ind_cost_ += self.connector.get_ind_cost(sql, indexes)
                    no_cost.append(no_cost_)
                    ind_cost.append(ind_cost_)
                elif self.args.cost_estimator == "model":
                    cost_ratio_ = 0
                    for sql in work_list:
                        cost_ratio_ += self.get_learned_cost_ratio(sql, indexes, "")
                    cost_ratio.append(cost_ratio_)
            else:
                if self.args.cost_estimator == "optimizer":
                    if self.args.work_level == "query":
                        no_cost.append(sql_token[self.args.victim].get("no_cost", 0))
                        ind_cost.append(sql_token[self.args.victim].get("ind_cost", 0))
                    elif self.args.work_level == "workload":
                        no_cost.append(sql_token["workload"][self.args.victim].get("no_cost", 0))
                        ind_cost.append(sql_token["workload"][self.args.victim].get("ind_cost", 0))
                    else:
                        raise NotImplementedError
                elif self.args.cost_estimator == "model":
                    if self.args.work_level == "query":
                        work_list = [sql_token["sql"]]
                        indexes = sql_token[self.args.victim]["indexes"]
                    elif self.args.work_level == "workload":
                        work_list = [sql_token["sql"]] + sql_token["workload"]["sql"]
                        indexes = sql_token["workload"][self.args.victim]["indexes"]

                    cost_ratio_ = 0
                    for sql in work_list:
                        cost_ratio_ += self.get_learned_cost_ratio(sql, indexes, "")
                    cost_ratio.append(cost_ratio_)

            if self.args.work_level == "query":
                work_list = [tgt_sql]
            elif self.args.work_level == "workload":
                work_list = [tgt_sql] + sql_token["workload"]["sql"]

            try:
                if "dynamic" in self.args.reward:
                    indexes = self.get_heur_indexes(self.args.victim, self.exp_conf, work_list)
                elif self.args.reward == "static":
                    if self.args.work_level == "query":
                        indexes = sql_token[self.args.victim]["indexes"]
                    elif self.args.work_level == "workload":
                        indexes = sql_token["workload"][self.args.victim]["indexes"]

                if self.args.cost_estimator == "optimizer":
                    no_cost_, ind_cost_ = 0, 0
                    for sql in work_list:
                        no_cost_ += self.connector.get_ind_cost(sql, "")
                        ind_cost_ += self.connector.get_ind_cost(sql, indexes)
                    no_cost.append(no_cost_)
                    ind_cost.append(ind_cost_)
                elif self.args.cost_estimator == "model":
                    cost_ratio_ = 0
                    for sql in work_list:
                        cost_ratio_ += self.get_learned_cost_ratio(sql, indexes, "")
                    cost_ratio.append(cost_ratio_)
            except Exception as e:
                if self.args.cost_estimator == "optimizer":
                    no_cost.append(no_cost[0])
                    ind_cost.append(ind_cost[0])
                elif self.args.cost_estimator == "model":
                    cost_ratio.append(1.)
                logging.error(e)
                logging.error(traceback.format_exc())

        # 2) base: comparison against the `baseline heuristic` case.
        else:
            # get the index recommendation result (src_workload) before perturbation from the scratch.
            if self.args.reward == "all_dynamic":
                if self.args.work_level == "query":
                    work_list = [sql_token["sql"]]
                elif self.args.work_level == "workload":
                    work_list = [sql_token["sql"]] + sql_token["workload"]["sql"]

                base_indexes = self.get_heur_indexes(self.args.baseline, self.base_exp_conf, work_list)
                indexes = self.get_heur_indexes(self.args.victim, self.exp_conf, work_list)

                if self.args.cost_estimator == "optimizer":
                    no_cost_, ind_cost_ = 0, 0
                    for sql in work_list:
                        no_cost_ += self.connector.get_ind_cost(sql, base_indexes)
                        ind_cost_ += self.connector.get_ind_cost(sql, indexes)
                    no_cost.append(no_cost_)
                    ind_cost.append(ind_cost_)
                elif self.args.cost_estimator == "model":
                    cost_ratio_ = 0
                    for sql in work_list:
                        cost_ratio_ += self.get_learned_cost_ratio(sql, indexes, base_indexes)
                    cost_ratio.append(cost_ratio_)
            else:
                if self.args.cost_estimator == "optimizer":
                    if self.args.work_level == "query":
                        no_cost.append(sql_token[self.args.baseline].get("ind_cost", 0))
                        ind_cost.append(sql_token[self.args.victim].get("ind_cost", 0))
                    elif self.args.work_level == "workload":
                        no_cost.append(sql_token["workload"][self.args.baseline].get("ind_cost", 0))
                        ind_cost.append(sql_token["workload"][self.args.victim].get("ind_cost", 0))
                    else:
                        raise NotImplementedError
                elif self.args.cost_estimator == "model":
                    if self.args.work_level == "query":
                        work_list = [sql_token["sql"]]
                        base_indexes = sql_token[self.args.baseline]["indexes"]
                        indexes = sql_token[self.args.victim]["indexes"]
                    elif self.args.work_level == "workload":
                        work_list = [sql_token["sql"]] + sql_token["workload"]["sql"]
                        base_indexes = sql_token["workload"][self.args.baseline]["indexes"]
                        indexes = sql_token["workload"][self.args.victim]["indexes"]

                    cost_ratio_ = 0
                    for sql in work_list:
                        cost_ratio_ += self.get_learned_cost_ratio(sql, indexes, base_indexes)
                    cost_ratio.append(cost_ratio_)

            if self.args.work_level == "query":
                work_list = [tgt_sql]
            elif self.args.work_level == "workload":
                work_list = [tgt_sql] + sql_token["workload"]["sql"]

            try:
                if "dynamic" in self.args.reward:
                    base_indexes = self.get_heur_indexes(self.args.baseline, self.base_exp_conf, work_list)
                    indexes = self.get_heur_indexes(self.args.victim, self.exp_conf, work_list)
                elif self.args.reward == "static":
                    if self.args.work_level == "query":
                        base_indexes = sql_token[self.args.baseline]["indexes"]
                        indexes = sql_token[self.args.victim]["indexes"]
                    elif self.args.work_level == "workload":
                        base_indexes = sql_token["workload"][self.args.baseline]["indexes"]
                        indexes = sql_token["workload"][self.args.victim]["indexes"]

                if self.args.cost_estimator == "optimizer":
                    no_cost_, ind_cost_ = 0, 0
                    for sql in work_list:
                        no_cost_ += self.connector.get_ind_cost(sql, base_indexes)
                        ind_cost_ += self.connector.get_ind_cost(sql, indexes)
                    no_cost.append(no_cost_)
                    ind_cost.append(ind_cost_)
                elif self.args.cost_estimator == "model":
                    cost_ratio_ = 0
                    for sql in work_list:
                        cost_ratio_ += self.get_learned_cost_ratio(sql, indexes, base_indexes)
                    cost_ratio.append(cost_ratio_)
            except Exception as e:
                if self.args.cost_estimator == "optimizer":
                    no_cost.append(no_cost[0])
                    ind_cost.append(ind_cost[0])
                elif self.args.cost_estimator == "model":
                    cost_ratio.append(1.)
                logging.error(e)
                logging.error(traceback.format_exc())

        if self.args.cost_estimator == "optimizer":
            return no_cost, ind_cost
        elif self.args.cost_estimator == "model":
            return cost_ratio

    def get_rl_perform(self, work_list):
        if "swirl" in self.args.victim:
            perform = get_swirl_res(self.swirl_exp, work_list, self.swirl_model)[0]
        elif "drlindex" in self.args.victim:
            perform = get_drlindex_res(self.swirl_exp, work_list, self.swirl_model)[0]
        elif "dqn" in self.args.victim:
            perform = get_dqn_res(self.swirl_exp, work_list, self.swirl_model)[0]

        return perform

    def get_rl_result(self, src_token, tgt_sql):
        cost_ratio = list()
        no_cost, ind_cost = list(), list()
        # 1) basic: comparison against the `without index` case.
        if not self.args.reward_base:
            # get the index recommendation result (src_workload) from the scratch before perturbation.
            if self.args.reward == "all_dynamic":
                if self.args.work_level == "query":
                    work_list = [src_token["sql"]]
                elif self.args.work_level == "workload":
                    work_list = [src_token["sql"]] + src_token["workload"]["sql"]

                perform = self.get_rl_perform(work_list)

                if self.args.cost_estimator == "optimizer":
                    no_cost.append(perform["no_cost"])
                    ind_cost.append(perform["ind_cost"])
                elif self.args.cost_estimator == "model":
                    cost_ratio_ = 0
                    for sql in work_list:
                        cost_ratio_ += self.get_learned_cost_ratio(sql, perform["indexes"], "")
                    cost_ratio.append(cost_ratio_)
            else:
                if self.args.cost_estimator == "optimizer":
                    # no_cost.append(src_token[self.args.victim].get("no_cost", 0))
                    # ind_cost.append(src_token[self.args.victim].get("ind_cost", 0))
                    if self.args.work_level == "query":
                        no_cost.append(src_token[self.args.victim].get("no_cost", 0))
                        ind_cost.append(src_token[self.args.victim].get("ind_cost", 0))
                    elif self.args.work_level == "workload":
                        no_cost.append(src_token["workload"][self.args.victim].get("no_cost", 0))
                        ind_cost.append(src_token["workload"][self.args.victim].get("ind_cost", 0))
                elif self.args.cost_estimator == "model":
                    if self.args.work_level == "query":
                        work_list = [src_token["sql"]]
                        indexes = src_token[self.args.victim]["indexes"]
                    elif self.args.work_level == "workload":
                        work_list = [src_token["sql"]] + src_token["workload"]["sql"]
                        indexes = src_token["workload"][self.args.victim]["indexes"]

                    cost_ratio_ = 0
                    for sql in work_list:
                        cost_ratio_ += self.get_learned_cost_ratio(sql, indexes, "")
                    cost_ratio.append(cost_ratio_)

            if self.args.work_level == "query":
                work_list = [tgt_sql]
            elif self.args.work_level == "workload":
                work_list = [tgt_sql] + src_token["workload"]["sql"]

            try:
                if "dynamic" in self.args.reward:
                    # logging.disable(logging.INFO)
                    perform = self.get_rl_perform(work_list)
                    # logging.disable(logging.DEBUG)

                    if self.args.cost_estimator == "optimizer":
                        no_cost.append(perform["no_cost"])
                        ind_cost.append(perform["ind_cost"])
                    elif self.args.cost_estimator == "model":
                        cost_ratio_ = 0
                        for sql in work_list:
                            cost_ratio_ += self.get_learned_cost_ratio(sql, perform["indexes"], "")
                        cost_ratio.append(cost_ratio_)
                elif self.args.reward == "static":
                    if self.args.work_level == "query":
                        indexes = src_token[self.args.victim]["indexes"]
                    elif self.args.work_level == "workload":
                        indexes = src_token["workload"][self.args.victim]["indexes"]

                    if self.args.cost_estimator == "optimizer":
                        no_cost_, ind_cost_ = 0, 0
                        for sql in work_list:
                            no_cost_ += self.connector.get_ind_cost(sql, "")
                            ind_cost_ += self.connector.get_ind_cost(sql, indexes)
                        no_cost.append(no_cost_)
                        ind_cost.append(ind_cost_)

                    elif self.args.cost_estimator == "model":
                        cost_ratio_ = 0
                        for sql in work_list:
                            cost_ratio_ += self.get_learned_cost_ratio(sql, indexes, "")
                        cost_ratio.append(cost_ratio_)
            except Exception as e:
                if self.args.cost_estimator == "optimizer":
                    no_cost.append(no_cost[0])
                    ind_cost.append(ind_cost[0])
                elif self.args.cost_estimator == "model":
                    cost_ratio.append(1)
                logging.error(e)
                logging.error(traceback.format_exc())

        # 2) base: comparison against the `baseline heuristic` case.
        else:
            # get the index recommendation result (src_workload) before perturbation from the scratch.
            if self.args.reward == "all_dynamic":
                if self.args.work_level == "query":
                    work_list = [src_token["sql"]]
                elif self.args.work_level == "workload":
                    work_list = [src_token["sql"]] + src_token["workload"]["sql"]

                indexes = self.get_heur_indexes(self.args.baseline, self.base_exp_conf, work_list)
                perform = self.get_rl_perform(work_list)
                if self.args.cost_estimator == "optimizer":
                    no_cost_ = 0
                    for sql in work_list:
                        no_cost_ += self.connector.get_ind_cost(sql, indexes)
                    no_cost.append(no_cost_)
                    ind_cost.append(perform["ind_cost"])
                elif self.args.cost_estimator == "model":
                    cost_ratio_ = 0
                    for sql in work_list:
                        cost_ratio_ += self.get_learned_cost_ratio(sql, perform["indexes"], indexes)
                    cost_ratio.append(cost_ratio_)
            else:
                if self.args.cost_estimator == "optimizer":
                    if self.args.work_level == "query":
                        no_cost.append(src_token[self.args.baseline].get("ind_cost", 0))
                        ind_cost.append(src_token[self.args.victim].get("ind_cost", 0))
                    elif self.args.work_level == "workload":
                        no_cost.append(src_token["workload"][self.args.baseline].get("ind_cost", 0))
                        ind_cost.append(src_token["workload"][self.args.victim].get("ind_cost", 0))

                elif self.args.cost_estimator == "model":
                    if self.args.work_level == "query":
                        work_list = [src_token["sql"]]
                        base_indexes = src_token[self.args.baseline]["indexes"]
                        indexes = src_token[self.args.victim]["indexes"]
                    elif self.args.work_level == "workload":
                        work_list = [src_token["sql"]] + src_token["workload"]["sql"]
                        base_indexes = src_token["workload"][self.args.baseline]["indexes"]
                        indexes = src_token["workload"][self.args.victim]["indexes"]

                    cost_ratio_ = 0
                    for sql in work_list:
                        cost_ratio_ += self.get_learned_cost_ratio(sql, indexes, base_indexes)
                    cost_ratio.append(cost_ratio_)

            if self.args.work_level == "query":
                work_list = [tgt_sql]
            elif self.args.work_level == "workload":
                work_list = [tgt_sql] + src_token["workload"]["sql"]

            try:
                if "dynamic" in self.args.reward:
                    # a) get the newly recommended index for the `heuristic baseline`.
                    indexes = self.get_heur_indexes(self.args.baseline, self.base_exp_conf, work_list)
                    # the index impact under `heuristic baseline` of the `tgt_sql` (perturbed sql).

                    # b) get the newly recommended index for the `rl-based vicitm`.
                    # logging.disable(logging.INFO)
                    perform = self.get_rl_perform(work_list)
                    # logging.disable(logging.DEBUG)

                    if self.args.cost_estimator == "optimizer":
                        no_cost_ = 0
                        for sql in work_list:
                            no_cost_ += self.connector.get_ind_cost(sql, indexes)
                        no_cost.append(no_cost_)
                        # no_cost.append(self.connector.get_ind_cost(tgt_sql, indexes))
                        # the index impact under `rl-based victim` of the `tgt_sql` (perturbed sql).
                        ind_cost.append(perform["ind_cost"])
                    elif self.args.cost_estimator == "model":
                        cost_ratio_ = 0
                        for sql in work_list:
                            cost_ratio_ += self.get_learned_cost_ratio(sql, perform["indexes"], indexes)
                        cost_ratio.append(cost_ratio_)

                elif self.args.reward == "static":
                    if self.args.work_level == "query":
                        base_indexes = src_token[self.args.baseline]["indexes"]
                        indexes = src_token[self.args.victim]["indexes"]
                    elif self.args.work_level == "workload":
                        base_indexes = src_token["workload"][self.args.baseline]["indexes"]
                        indexes = src_token["workload"][self.args.victim]["indexes"]

                    if self.args.cost_estimator == "optimizer":
                        no_cost_, ind_cost_ = 0, 0
                        for sql in work_list:
                            no_cost_ += self.connector.get_ind_cost(sql, base_indexes)
                            ind_cost_ += self.connector.get_ind_cost(sql, indexes)
                        no_cost.append(no_cost_)
                        ind_cost.append(ind_cost_)
                    elif self.args.cost_estimator == "model":
                        cost_ratio_ = 0
                        for sql in work_list:
                            cost_ratio_ += self.get_learned_cost_ratio(sql, indexes, base_indexes)
                        cost_ratio.append(cost_ratio_)
            except Exception as e:
                if self.args.cost_estimator == "optimizer":
                    no_cost.append(no_cost[0])
                    ind_cost.append(ind_cost[0])
                elif self.args.cost_estimator == "model":
                    cost_ratio.append(1)
                logging.error(e)
                logging.error(src_token["sql"])
                logging.error(tgt_sql)
                logging.error(traceback.format_exc())

        if self.args.cost_estimator == "optimizer":
            return no_cost, ind_cost
        elif self.args.cost_estimator == "model":
            return cost_ratio

    def get_index_reward(self, src_token, tgt_vec, idx2word, col_info):
        tgt_sql = vec2sql([src_token], [tgt_vec], idx2word, col_info)[0]["sql_text"]

        if self.args.cost_estimator == "optimizer":
            if self.args.victim in constants.heuristic:
                no_cost, ind_cost = self.get_heur_result(src_token, tgt_sql)
            elif self.args.victim in constants.rl_based or \
                    "mcts" in self.args.victim:
                no_cost, ind_cost = self.get_rl_result(src_token, tgt_sql)
            else:
                raise NotImplemented
        elif self.args.cost_estimator == "model":
            if self.args.victim in constants.heuristic:
                cost_ratio = self.get_heur_result(src_token, tgt_sql)
            elif self.args.victim in constants.rl_based or \
                    "mcts" in self.args.victim:
                cost_ratio = self.get_rl_result(src_token, tgt_sql)
            else:
                raise NotImplemented

        if self.args.cost_estimator == "optimizer":
            if no_cost[0] == 0 or no_cost[1] == 0:
                reward = 0.
            else:
                if self.args.reward_form == "cost_red_ratio":
                    before, after = 1 - ind_cost[0] / no_cost[0], 1 - ind_cost[1] / no_cost[1]
                    if before <= 0:
                        reward = 0.
                    else:
                        reward = 1 - after / before
                elif self.args.reward_form == "cost_ratio":
                    before, after = ind_cost[0] / no_cost[0], ind_cost[1] / no_cost[1]
                    reward = (after / before)
                elif self.args.reward_form == "cost_ratio_norm":
                    before, after = ind_cost[0] / no_cost[0], ind_cost[1] / no_cost[1]
                    reward = (after / before) - 1.0
                elif self.args.reward_form == "inv_cost_ratio_norm":
                    if ind_cost[1] == 0 or ind_cost[0] == 0:
                        reward = 0.
                    else:
                        before, after = no_cost[0] / ind_cost[0], no_cost[1] / ind_cost[1]
                        reward = 1.0 - (after / before)
        if self.args.cost_estimator == "model":
            if self.args.reward_form == "cost_red_ratio":
                before, after = 1 - cost_ratio[0], 1 - cost_ratio[1]
                if before <= 0:
                    reward = 0.
                else:
                    reward = 1 - after / before
            elif self.args.reward_form == "cost_ratio":
                before, after = cost_ratio[0], cost_ratio[1]
                reward = (after / before)
            elif self.args.reward_form == "cost_ratio_norm":
                before, after = cost_ratio[0], cost_ratio[1]
                reward = (after / before) - 1.0

        return reward
