import argparse
import itertools
import json
import logging

from heuristic_advisor.heuristic_utils.index import Index
from heuristic_advisor.heuristic_utils.postgres_dbms import PostgresDatabaseConnector
from heuristic_advisor.heuristic_utils.workload import Workload

from learning_advisor.learning_utils.cost_evaluation import CostEvaluation


def get_parser():
    parser = argparse.ArgumentParser(
        description="the testbed of Learning-based Index Advisors.")

    parser.add_argument("--exp_id", type=str, default="swirl_exp_id")
    parser.add_argument("--victim", type=str, default="swirl",
                        choices=["swirl", "drlindex", "dqn"])
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--max_indexes", type=int, default=5)

    parser.add_argument("--exp_conf_file", type=str,
                        default="./data_resource/learning_conf/swirl_config.json")
    parser.add_argument("--db_conf_file", type=str,
                        default="./data_resource/database_conf/db_info.conf")
    parser.add_argument("--schema_file", type=str,
                        default="./data_resource/database_conf/schema.json")
    parser.add_argument("--colinfo_load", type=str,
                        default="./data_resource/database_conf/colinfo.json")

    parser.add_argument("--work_type", type=str, default="not_template")
    parser.add_argument("--work_file", type=str,
                        default="./data_resource/sample_data/sample_query.sql")
    parser.add_argument("--temp_expand", action="store_true")
    parser.add_argument("--temp_load", type=str,
                        default="./data_resource/sample_data/sample_token.json")

    parser.add_argument("--train_mode", type=str, default="scratch",
                        choices=["continuous", "scratch"])
    parser.add_argument("--rl_exp_load", type=str,
                        default="")
    parser.add_argument("--rl_model_load", type=str,
                        default="")
    parser.add_argument("--rl_env_load", type=str,
                        default="")

    parser.add_argument("--res_save_path", type=str, default="./exp_res")
    parser.add_argument("--logdir", type=str,
                        default="./exp_res/{}/logdir")
    parser.add_argument("--log_file", type=str,
                        default="./exp_res/{}/exp_runtime.log")

    return parser


def set_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
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


def predict_index_sizes(column_combinations, db_config, is_precond=True):
    connector = PostgresDatabaseConnector(db_config, autocommit=True)
    connector.drop_indexes()

    cost_evaluation = CostEvaluation(connector)

    predicted_index_sizes = []
    parent_index_size_map = {}
    for column_combination in column_combinations:
        potential_index = Index(column_combination)
        cost_evaluation.what_if.simulate_index(potential_index, store_size=True)

        full_index_size = potential_index.estimated_size
        index_delta_size = full_index_size

        if is_precond:
            if len(column_combination) > 1:
                index_delta_size -= parent_index_size_map[column_combination[:-1]]

        predicted_index_sizes.append(index_delta_size)
        cost_evaluation.what_if.drop_simulated_index(potential_index)

        parent_index_size_map[column_combination] = full_index_size

    connector.close()

    return predicted_index_sizes


def get_hypo_index_sizes(column_combinations, db_config):
    connector = PostgresDatabaseConnector(db_config, autocommit=True)
    connector.drop_indexes()

    cost_evaluation = CostEvaluation(connector)

    predicted_index_sizes = []
    parent_index_size_map = {}
    for column_combination in column_combinations:
        potential_index = Index(column_combination)
        cost_evaluation.what_if.simulate_index(potential_index, store_size=True)

        full_index_size = potential_index.estimated_size
        index_delta_size = full_index_size
        if len(column_combination) > 1 and column_combination[:-1] in parent_index_size_map.keys():
            index_delta_size -= parent_index_size_map[column_combination[:-1]]

        predicted_index_sizes.append(index_delta_size)
        cost_evaluation.what_if.drop_simulated_index(potential_index)

        parent_index_size_map[column_combination] = full_index_size

    connector.close()

    return predicted_index_sizes


def get_prom_index_candidates(token_load, colinfo_load, columns):
    result_column_combinations = list()

    with open(token_load, "r") as rf:
        sql_tokens = json.load(rf)
    with open(colinfo_load, "r") as rf:
        col_info = json.load(rf)

    column_dict = dict()  # w_warehouse_sk
    for col in columns:
        column_dict[str(col).split(".")[-1]] = col

    table_column_dict = dict()
    for column in columns:
        if column.table not in table_column_dict:
            table_column_dict[column.table] = set()
        table_column_dict[column.table].add(column)

    prom_ind = {1: list()}
    # all the single-column index.
    for col in column_dict.values():
        prom_ind[1].append(tuple([col]))
    for sql_token in sql_tokens:
        # 1. extract the columns in certain positions.
        join_col = list()
        for typ, tok in zip(sql_token["from"]["pre_type"], sql_token["from"]["pre_token"]):
            if typ == "from#join_column" and tok.split(".")[-1] not in join_col \
                    and tok.split(".")[-1] in column_dict.keys():
                join_col.append(tok.split(".")[-1])

        eq_col, range_col = list(), list()
        for i, tok in enumerate(sql_token["where"]["pre_token"]):
            if tok == "=" and \
                    sql_token["where"]["pre_token"][i - 1].split(".")[-1] not in eq_col \
                    and sql_token["where"]["pre_token"][i - 1].split(".")[-1] in column_dict.keys():
                eq_col.append(sql_token["where"]["pre_token"][i - 1].split(".")[-1])
            if tok in [">", "<", ">=", "<="] and \
                    sql_token["where"]["pre_token"][i - 1].split(".")[-1] not in range_col \
                    and sql_token["where"]["pre_token"][i - 1].split(".")[-1] in column_dict.keys():
                range_col.append(sql_token["where"]["pre_token"][i - 1].split(".")[-1])

        gro_col = list()
        if "group by" in sql_token.keys():
            for typ, tok in zip(sql_token["group by"]["pre_type"], sql_token["group by"]["pre_token"]):
                if typ == "group by#column" and tok.split(".")[-1] in column_dict.keys():
                    gro_col.append(tok.split(".")[-1])

        ord_col = list()
        if "order by" in sql_token.keys():
            for typ, tok in zip(sql_token["order by"]["pre_type"], sql_token["order by"]["pre_token"]):
                if typ == "order by#column" and tok.split(".")[-1] in column_dict.keys():
                    ord_col.append(tok.split(".")[-1])

        # 2. get the promising index combinations.
        gro_tbl = list(set([col_info[col]["table"] for col in gro_col]))
        if len(gro_tbl) == 1:
            ind = tuple([column_dict[col] for col in gro_col])
            if len(ind) not in prom_ind.keys():
                prom_ind[len(ind)] = list()
            if ind not in prom_ind[len(ind)]:
                prom_ind[len(ind)].append(ind)
        ord_tbl = list(set([col_info[col]["table"] for col in ord_col]))
        if len(ord_tbl) == 1:
            ind = tuple([column_dict[col] for col in ord_col])
            if len(ind) not in prom_ind.keys():
                prom_ind[len(ind)] = list()
            if ind not in prom_ind[len(ind)]:
                prom_ind[len(ind)].append(ind)

        join_tbl = list()
        for col in join_col:
            tbl = col_info[col]["table"]
            if tbl in join_tbl:
                continue
            join_tbl.append(tbl)

            ind = tuple([column_dict[col] for col in join_col if col_info[col]['table'] == tbl])
            if len(ind) not in prom_ind.keys():
                prom_ind[len(ind)] = list()
            if ind not in prom_ind[len(ind)]:
                prom_ind[len(ind)].append(ind)

        j_EQ_r_tbl = list()
        for j in join_col:
            tbl = col_info[j]["table"]
            if tbl in j_EQ_r_tbl:
                continue

            # multiple join columns from the same table.
            j = [column_dict[col] for col in join_col if col_info[col]["table"] == tbl]
            EQ = [column_dict[col] for col in eq_col if col_info[col]["table"] == tbl]
            r = [column_dict[col] for col in range_col if col_info[col]["table"] == tbl]

            ind = tuple(j + EQ + r)
            if len(ind) not in prom_ind.keys():
                prom_ind[len(ind)] = list()
            if ind not in prom_ind[len(ind)]:
                prom_ind[len(ind)].append(ind)

        EQ_r_tbl = list()
        for EQ in eq_col:
            tbl = col_info[EQ]["table"]
            if tbl in EQ_r_tbl:
                continue

            r = [column_dict[col] for col in range_col if col_info[col]["table"] == tbl]

            ind = tuple([column_dict[EQ]] + r)
            if len(ind) not in prom_ind.keys():
                prom_ind[len(ind)] = list()
            if ind not in prom_ind[len(ind)]:
                prom_ind[len(ind)].append(ind)

        j_r_tbl = list()
        for j in join_col:
            tbl = col_info[j]["table"]
            if tbl in j_r_tbl:
                continue

            # multiple join columns from the same table.
            j = [column_dict[col] for col in join_col if col_info[col]["table"] == tbl]
            r = [column_dict[col] for col in range_col if col_info[col]["table"] == tbl]

            ind = tuple(j + r)
            if len(ind) not in prom_ind.keys():
                prom_ind[len(ind)] = list()
            if ind not in prom_ind[len(ind)]:
                prom_ind[len(ind)].append(ind)

        j_EQ_tbl = list()
        for j in join_col:
            tbl = col_info[j]["table"]
            if tbl in j_EQ_tbl:
                continue

            j = [column_dict[col] for col in join_col if col_info[col]["table"] == tbl]
            EQ = [column_dict[col] for col in eq_col if col_info[col]["table"] == tbl]

            ind = tuple(j + EQ)
            if len(ind) not in prom_ind.keys():
                prom_ind[len(ind)] = list()
            if ind not in prom_ind[len(ind)]:
                prom_ind[len(ind)].append(ind)

    for key in sorted(prom_ind.keys()):
        result_column_combinations.append(prom_ind[key])

    return result_column_combinations


def create_column_permutation_indexes(columns, max_index_width):
    result_column_combinations = []

    table_column_dict = {}
    for column in columns:
        if column.table not in table_column_dict:
            table_column_dict[column.table] = set()
        table_column_dict[column.table].add(column)

    for length in range(1, max_index_width + 1):
        unique = set()
        count = 0
        for key, columns_per_table in table_column_dict.items():
            unique |= set(itertools.permutations(columns_per_table, length))  # permutation: the orders that matter.
            count += len(set(itertools.permutations(columns_per_table, length)))
        logging.info(f"the total number of the unique {length}-column indexes is: {count}")

        result_column_combinations.append(sorted(list(unique)))

    return result_column_combinations


def get_utilized_indexes(
    workload, indexes_per_query, cost_evaluation, detailed_query_information=False
):
    utilized_indexes_workload = set()
    query_details = {}
    for query, indexes in zip(workload.queries, indexes_per_query):
        (
            utilized_indexes_query,
            cost_with_indexes,
        ) = cost_evaluation.which_indexes_utilized_and_cost(query, indexes)
        utilized_indexes_workload |= utilized_indexes_query

        if detailed_query_information:
            cost_without_indexes = cost_evaluation.calculate_cost(
                Workload([query]), indexes=[]
            )

            query_details[query] = {
                "cost_without_indexes": cost_without_indexes,
                "cost_with_indexes": cost_with_indexes,
                "utilized_indexes": utilized_indexes_query,
            }

    return utilized_indexes_workload, query_details


# Storage
def b_to_mb(b):
    return b / 1000 / 1000


def mb_to_b(mb):
    return mb * 1000 * 1000


# Time
def s_to_ms(s):
    return s * 1000
