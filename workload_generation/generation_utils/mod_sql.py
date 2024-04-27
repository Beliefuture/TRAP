import configparser
import copy
import json
import random
import logging

import sqlparse
import mo_sql_parsing as mosqlparse

import traceback
import numpy as np

# import constants
from tqdm import tqdm

from workload_generation.generation_utils import constants


random.seed(666)
perturb_prop = 1  # 0.4  # < perturb_prop, then random.choice(cand)

# file_prefix = ""
#
# conf_file = f"../data_resource/db_info/db.conf"
#
# word2idx_json = f"../data_resource/visrew_vocab/word2idx.json"
# idx2word_json = f"../data_resource/visrew_vocab/idx2word.json"
#
# colinfo_json = f"../data_resource/db_info/format_pre_col_tpch_1gb.json"
# wordinfo_json = f"../data_resource/visrew_vocab/word_info.json"


# with open(word2idx_json, "r") as rf:
#     word2idx = json.load(rf)
# with open(idx2word_json, "r") as rf:
#     idx2word = json.load(rf)
#
# with open(colinfo_json, "r") as rf:
#     col_info = json.load(rf)
# with open(wordinfo_json, "r") as rf:
#     word_info = json.load(rf)


def valid_cand(token, table, step, ptok_nos, word2idx,
               idx2word, word_info, col_info, max_diff=5):
    """

    :param token: dict
    :param table: list(str), table names
    :param step: int, current decoded time-step
    :param ptok_nos: list(int), tokens decoded already
    :param word2idx:
    :param idx2word:
    :param word_info:
    :param col_info:
    :param max_diff:
    :return:
    """

    if step >= len(token["pre_types"]):
        return [constants.PAD]

    cand = list()
    # 1) reserved: grammar keyword
    if token["pre_types"][step] in constants.keyword:
        cand = [word2idx[token["pre_tokens"][step].lower()]]
    elif token["pre_types"][step].upper() in constants.join:
        cand = [word2idx[token["pre_tokens"][step]]]

    # 2) tables
    elif "#join_table" in token["pre_types"][step]:
        cand = [word2idx[token["pre_tokens"][step]]]
    elif "#table" in token["pre_types"][step]:
        cand = [word2idx[tbl] for tbl in table if word2idx[tbl] not in ptok_nos]

    # 3) columns
    # elif "from#column" in token["pre_types"][i]:
    elif "#join_column" in token["pre_types"][step]:  # from/where
        cand = [word2idx[token["pre_tokens"][step].split(".")[-1]]]
    elif token["pre_types"][step] == "select#aggregate_column":
        for tbl in table:
            tbl_col = list(range(word_info[f"{tbl}#column name"]["start_id"],
                                 word_info[f"{tbl}#column name"]["end_id"] + 1))
            # 3.1) max()/min(): column of all types.
            if idx2word[str(ptok_nos[-1])] in constants.aggregator[:2]:
                cand.extend(tbl_col)
            # 3.2) count()/avg()/sum(): column of numeric types.
            # elif idx2word[str(ptok_nos[-1])] in constants.aggregator[-3:]:
            elif idx2word[str(ptok_nos[-1])] in constants.aggregator[-3:]:
                cand.extend([col for col in tbl_col
                             if col_info[idx2word[str(col)]]["type"]
                             in ["integer", "numeric"]])
        # todo: numeric aggregate column can be the same, filter columns applied under the same aggregator selected already.
        cand_bak = copy.deepcopy(cand)
        cand = list(set(cand) - set([no for i, no in enumerate(ptok_nos)
                                     if token["pre_types"][i] == token["pre_types"][step]]))
        if len(cand) == 0:  # not enough numeric column
            cand = list(set(cand_bak) - set([no for i, no in enumerate(ptok_nos)
                                             if token["pno_tokens"][i - 1] == token["pno_tokens"][step - 1]]))
    # todo: special column (group by)
    elif token["pre_types"][step] == "group by#column":
        cand = list(set([no for i, no in enumerate(ptok_nos) if token["pre_types"][i] == "select#column"])
                    - set([no for i, no in enumerate(ptok_nos) if token["pre_types"][i] == "group by#column"]))
    # todo: special column (having)
    elif token["pre_types"][step] == "having#aggregate_column":
        cand = list(set([tok_no for i, tok_no in enumerate(ptok_nos) if
                         i >= 1 and token["pre_types"][i - 1] == "select#aggregator" and
                         ptok_nos[i - 1] == ptok_nos[step - 1]])
                    # - set([no for i, no in enumerate(ptok_nos) if
                    #        token["pre_types"][i] == "having#aggregate_column"]))
                    - set([no for i, no in enumerate(ptok_nos) if
                           token["pre_types"][i] == "having#aggregate_column" and
                           ptok_nos[i - 1] == ptok_nos[step - 1]]))
    # todo: special column (order by)
    elif token["pre_types"][step] == "order by#column":
        cand = list(set([no for i, no in enumerate(ptok_nos) if token["pre_types"][i] == "select#column"])
                    - set([no for i, no in enumerate(ptok_nos) if token["pre_types"][i] == "order by#column"]))
    elif token["pre_types"][step] == "order by#aggregate_column":
        cand = list(set([no for i, no in enumerate(ptok_nos) if
                         i >= 1 and token["pre_types"][i - 1] == "select#aggregator" and
                         ptok_nos[i - 1] == ptok_nos[step - 1]])
                    # - set([no for i, no in enumerate(ptok_nos) if
                    #        token["pre_types"][i] == "order by#aggregate_column"]))
                    - set([no for i, no in enumerate(ptok_nos) if
                           token["pre_types"][i] == "order by#aggregate_column" and
                           ptok_nos[i - 1] == ptok_nos[step - 1]]))
    # todo: repeated column
    elif "#column" in token["pre_types"][step]:  # select/where
        for tbl in table:
            cand.extend(list(range(word_info[f"{tbl}#column name"]["start_id"],
                                   word_info[f"{tbl}#column name"]["end_id"] + 1)))
        cand = list(set(cand) - set([no for i, no in enumerate(ptok_nos)
                                     if token["pre_types"][i] == token["pre_types"][step]]))
        # todo: to be removed.
        # try:
        #     cand = list(set(cand) - set([no for i, no in enumerate(ptok_nos)
        #                                  if token["pre_types"][i] == token["pre_types"][step]]))
        # except:
        #     print("#column")

    # 4) values
    # 4.1) common values: column values and min()/max() aggregate values.
    elif "#value" in token["pre_types"][step] or \
            "#aggregate_value" in token["pre_types"][step]:
        # cand = list(range(word_info[f"{idx2word[str(ptok_nos[-2])]}#column values"]["start_id"],
        #                   word_info[f"{idx2word[str(ptok_nos[-2])]}#column values"]["end_id"] + 1))
        try:
            cand = list(range(word_info[f"{idx2word[str(ptok_nos[-2])]}#column values"]["start_id"],
                              word_info[f"{idx2word[str(ptok_nos[-2])]}#column values"]["end_id"] + 1))
            sql_res = vec2sql([token], [ptok_nos], idx2word, col_info)[0]
            # sql_res["pre_tokens"] = token
            # sql_res["ptok_nos"] = ptok_nos.tolist()
            # with open("./test.json", "w") as wf:
            #     json.dump(sql_res, wf, indent=2)
        except:
            sql_res = vec2sql([token], [ptok_nos], idx2word, col_info)[0]
            sql_res["pre_tokens"] = token
            sql_res["ptok_nos"] = ptok_nos.tolist()
            with open("./except.json", "w") as wf:
                json.dump(sql_res, wf, indent=2)
            print(traceback.print_exc())
            print("#value/#aggregate_value")
    # elif "#aggregate_value" in token["pre_types"][tno]:
    #     cand = [-1]  # list(range(74, 3008))  # [-1]
    # 4.2) special values: count()/avg()/sum() numeric aggregate values.
    # only one.
    elif len(ptok_nos) >= 3 and idx2word[str(ptok_nos[-3])] in constants.aggregator[2:] and \
            "#numeric_aggregate_value" in token["pre_types"][step]:
        cand = [constants.UNK]  # UNK
    # more than one.
    elif "#numeric_aggregate_value" in token["pre_types"][step]:
        # cand = list(range(word_info[f"{idx2word[str(ptok_nos[-2])]}#column values"]["start_id"],
        #                   word_info[f"{idx2word[str(ptok_nos[-2])]}#column values"]["end_id"] + 1))
        try:
            cand = list(range(word_info[f"{idx2word[str(ptok_nos[-2])]}#column values"]["start_id"],
                              word_info[f"{idx2word[str(ptok_nos[-2])]}#column values"]["end_id"] + 1))
        except:
            print(traceback.format_exc())
            print("#numeric_aggregate_value")
    # elif "#numeric_aggregate_value" in token["pre_types"][step]:
    #     cand = [constants.UNK]  # UNK

    # 5) operations
    elif "#join_operator" in token["pre_types"][step]:
        cand = [word2idx[token["pre_tokens"][step]]]
    elif "operator" in token["pre_types"][step]:
        cand = list(range(word_info["operator"]["start_id"],
                          word_info["operator"]["end_id"] + 1))

    # todo: numeric_aggregate_value? min/max
    # elif "select#aggregator" in token["pre_types"][step]:
    #     cand = list(range(word_info["aggregator"]["start_id"],
    #                       word_info["aggregator"]["end_id"] + 1))
    # min/max
    elif token["pre_types"][step] == "select#aggregator" and \
            token["pre_tokens"][step].lower() in constants.aggregator[:2]:
        cand = list(range(word_info["aggregator"]["start_id"],
                          word_info["aggregator"]["start_id"] + 2))
    # todo: count/avg/sum, static?
    elif token["pre_types"][step] == "select#aggregator" and \
            token["pre_tokens"][step].lower() in constants.aggregator[2:]:
        cand = [word2idx[token["pre_tokens"][step].lower()]]
    # elif token["pre_types"][step] == "select#aggregator":
    #     cand = [word2idx[token["pre_tokens"][step].lower()]]

    elif token["pre_types"][step] == "having#aggregator":
        cand = [tok_no for i, tok_no in enumerate(ptok_nos)
                if token["pre_types"][i] == "select#aggregator"]
        existed = [tok_no for i, tok_no in enumerate(ptok_nos)
                   if token["pre_types"][i] == "having#aggregator"]
        # not a set, multiple aggregators in the select clause.
        for agg in existed:
            cand.remove(agg)
    elif "order by#aggregator" in token["pre_types"][step]:
        cand = [tok_no for i, tok_no in enumerate(ptok_nos)
                if token["pre_types"][i] == "select#aggregator"]
        existed = [tok_no for i, tok_no in enumerate(ptok_nos)
                   if token["pre_types"][i] == "order by#aggregator"]
        # not a set, multiple aggregators in the select clause.
        for agg in existed:
            cand.remove(agg)
    elif "conjunction" in token["pre_types"][step]:
        cand = list(range(word_info["conjunction"]["start_id"],
                          word_info["conjunction"]["end_id"] + 1))
    elif "order_by_key" in token["pre_types"][step]:
        cand = list(range(word_info["order by key"]["start_id"],
                          word_info["order by key"]["end_id"] + 1))

    # 6) predicate
    elif "null_predicate" in token["pre_types"][step]:
        cand = list(range(word_info["null"]["start_id"],
                          word_info["null"]["end_id"] + 1))
    elif "in_predicate" in token["pre_types"][step]:
        cand = list(range(word_info["in"]["start_id"],
                          word_info["in"]["end_id"] + 1))
    elif "exists_predicate" in token["pre_types"][step]:
        cand = list(range(word_info["exists"]["start_id"],
                          word_info["exists"]["end_id"] + 1))
    elif "like_predicate" in token["pre_types"][step]:
        cand = list(range(word_info["like"]["start_id"],
                          word_info["like"]["end_id"] + 1))

    # todo: perturbation step constraint, already decoded words difference (not forcibly).
    #  group by / order by / having clause.
    if np.sum(np.array(token["pno_tokens"][:step]) != np.array(ptok_nos)) >= max_diff \
            and token["pno_tokens"][step] in cand:
        return [token["pno_tokens"][step]]
    else:
        return cand


def valid_cand_col(token, table, step, ptok_nos, column_left, word2idx,
                   idx2word, word_info, col_info, max_diff=5):
    """

    :param token: dict
    :param table: list(str), table names
    :param step: int, current decoded time-step
    :param ptok_nos: list(int), tokens decoded already
    :param column_left : list(int), the left column candidates
    :param word2idx:
    :param idx2word:
    :param word_info:
    :param col_info:
    :param max_diff:
    :return:
    """

    # todo: time-step exceed the max_len.
    if step >= len(token["pre_types"]):
        return [constants.PAD]

    # todo(might not executable): perturbation step constraint, forcibly truncated.
    # if np.sum(np.array(src_vec[:step]) != np.array(ptok_nos)) >= max_diff \
    #         and src_vec[step] in cand:
    #     return [src_vec[step]]

    # only the value is associated with the column selected.
    if "#column" not in token["pre_types"][step] and \
            "#aggregate_column" not in token["pre_types"][step] and \
            "#value" not in token["pre_types"][step] and \
            "#aggregate_value" not in token["pre_types"][step] and \
            "#numeric_aggregate_value" not in token["pre_types"][step]:
        return [token["pno_tokens"][step]]

    cand = list()
    # 3) columns
    # todo: special column (aggregate)
    #  type for min()/avg()/count()
    if token["pre_types"][step] == "select#aggregate_column":
        # 3.1) max()/min(): column of all types.
        if idx2word[str(ptok_nos[-1])] in constants.aggregator[:2]:
            # if idx2word[str(ptok_nos[-1])] in constants.aggregator[:3]:
            cand.extend(column_left)
        # todo: count()
        # elif ptoken[-1] == constants.aggregator[2]:
        #     cand.extend([col for col in tbl_col
        #                  if col_info[idx2word[str(col)]]["type"] != "date"])
        # 3.2) count()/avg()/sum(): column of numeric types.
        # elif idx2word[str(ptok_nos[-1])] in constants.aggregator[-3:]:
        #     cand.extend([col for col in column_left
        #                  if col_info[idx2word[str(col)]]["type"]
        #                  in ["integer", "numeric"]])
        # filter columns selected in the same clause already.
        cand = list(set(cand) - set([no for i, no in enumerate(ptok_nos)
                                     if token["pre_types"][i] == token["pre_types"][step]]))
        # the only `["integer", "numeric"] #aggregate_column` has been chosen.
        if len(cand) == 0:
            cand = list(set([no for i, no in enumerate(token["pno_tokens"])
                             if token["pno_tokens"][i - 1] == token["pno_tokens"][step - 1]])
                        - set([no for i, no in enumerate(ptok_nos)
                               if ptok_nos[i - 1] == ptok_nos[step - 1]]))
            # cand.append(token["pno_tokens"][step])
    # todo: special column (group by)
    elif token["pre_types"][step] == "group by#column":
        cand = list(set([no for i, no in enumerate(ptok_nos) if token["pre_types"][i] == "select#column"])
                    - set([no for i, no in enumerate(ptok_nos) if token["pre_types"][i] == "group by#column"]))
    # todo: special column (having)
    elif token["pre_types"][step] == "having#aggregate_column":
        cand = list(set([tok_no for i, tok_no in enumerate(ptok_nos) if
                         i >= 1 and token["pre_types"][i - 1] == "select#aggregator" and
                         ptok_nos[i - 1] == ptok_nos[step - 1]])
                    - set([no for i, no in enumerate(ptok_nos) if
                           token["pre_types"][i] == "having#aggregate_column" and
                           ptok_nos[i - 1] == ptok_nos[step - 1]]))
    # todo: special column (order by)
    elif token["pre_types"][step] == "order by#column":
        cand = list(set([no for i, no in enumerate(ptok_nos) if token["pre_types"][i] == "select#column"])
                    - set([no for i, no in enumerate(ptok_nos) if token["pre_types"][i] == "order by#column"]))
    elif token["pre_types"][step] == "order by#aggregate_column":
        cand = list(set([no for i, no in enumerate(ptok_nos) if
                         i >= 1 and token["pre_types"][i - 1] == "select#aggregator" and
                         ptok_nos[i - 1] == ptok_nos[step - 1]])
                    - set([no for i, no in enumerate(ptok_nos) if
                           token["pre_types"][i] == "order by#aggregate_column" and
                           ptok_nos[i - 1] == ptok_nos[step - 1]]))
    # todo: repeated column
    elif "#column" in token["pre_types"][step]:  # select/where
        cand = list(set(column_left) - set([no for i, no in enumerate(ptok_nos)
                                            if token["pre_types"][i] == token["pre_types"][step]]))
        if len(cand) == 0 and token["pre_types"][step] == "where#column":
            cand = list(set([no for i, no in enumerate(ptok_nos)
                             if token["pre_types"][i] == "select#column"
                             or token["pre_types"][i] == "select#aggregate_column"])
                        - set([no for i, no in enumerate(ptok_nos)
                               if token["pre_types"][i] == "where#column"]))
        # cand = column_left  # todo: to be removed, for repeated columns.

    # 4) values
    # 4.1) common values: column values and min()/max() aggregate values.
    elif "#value" in token["pre_types"][step] or \
            "#aggregate_value" in token["pre_types"][step]:
        # cand = list(range(word_info[f"{idx2word[str(ptok_nos[-2])]}#column values"]["start_id"],
        #                   word_info[f"{idx2word[str(ptok_nos[-2])]}#column values"]["end_id"] + 1))
        try:
            cand = list(range(word_info[f"{idx2word[str(ptok_nos[-2])]}#column values"]["start_id"],
                              word_info[f"{idx2word[str(ptok_nos[-2])]}#column values"]["end_id"] + 1))
        except:
            vec2sql([token], [ptok_nos], idx2word, col_info)
            # print("#value/#aggregate_value")
    # elif "#aggregate_value" in token["pre_types"][tno]:
    #     cand = [-1]  # list(range(74, 3008))  # [-1]
    # 4.2) special values: count()/avg()/sum() numeric aggregate values.
    # only one.
    elif len(ptok_nos) >= 3 and idx2word[str(ptok_nos[-3])] in constants.aggregator[2:] and \
            "#numeric_aggregate_value" in token["pre_types"][step]:
        cand = [constants.UNK]  # UNK
    # more than one.
    elif "#numeric_aggregate_value" in token["pre_types"][step]:
        # cand = list(range(word_info[f"{idx2word[str(ptok_nos[-2])]}#column values"]["start_id"],
        #                   word_info[f"{idx2word[str(ptok_nos[-2])]}#column values"]["end_id"] + 1))
        try:
            cand = list(range(word_info[f"{idx2word[str(ptok_nos[-2])]}#column values"]["start_id"],
                              word_info[f"{idx2word[str(ptok_nos[-2])]}#column values"]["end_id"] + 1))
        except:
            print("#numeric_aggregate_value")

    try:
        assert len(cand) != 0, "The list of `cand` is empty."
    except:
        logging.error("cand empty error!")
        valid_cand_col(token, table, step, ptok_nos, column_left, word2idx,
                       idx2word, word_info, col_info, max_diff)
        sql_res = vec2sql([token], [ptok_nos], idx2word, col_info)[0]
        sql_res["pre_tokens"] = token
        sql_res["ptok_nos"] = ptok_nos.tolist()
        with open("./col_cand_except.json", "w") as wf:
            json.dump(sql_res, wf, indent=2)
        logging.error(traceback.print_exc())
        raise AssertionError

    # todo: perturbation step constraint, already decoded words difference (not forcibly).
    #  group by / order by / having clause.
    if np.sum(np.array(token["pno_tokens"][:step]) != np.array(ptok_nos)) >= max_diff \
            and token["pno_tokens"][step] in cand:
        return [token["pno_tokens"][step]]
    else:
        return cand


def valid_cand_val(token, table, step, ptok_nos, word2idx,
                   idx2word, word_info, col_info, max_diff=5):
    """

    :param token: dict
    :param table: list(str), table names
    :param step: int, current decoded time-step
    :param ptok_nos: list(int), tokens decoded already
    :param word2idx:
    :param idx2word:
    :param word_info:
    :param col_info:
    :param max_diff:
    :return:
    """
    # todo: time-step exceed the max_len.
    if step >= len(token["pre_types"]):
        return [constants.PAD]

    # todo(might not executable): perturbation step constraint, forcibly.
    # if np.sum(np.array(src_vec[:step]) != np.array(ptok_nos)) >= max_diff \
    #         and src_vec[step] in cand:
    #     return [src_vec[step]]

    if "#value" not in token["pre_types"][step] and \
            "#aggregate_value" not in token["pre_types"][step] and \
            "#numeric_aggregate_value" not in token["pre_types"][step]:
        return [token["pno_tokens"][step]]

    cand = list()
    # 1) values
    # 1.1) common values: column values and min()/max() aggregate values.
    if "#value" in token["pre_types"][step] or \
            "#aggregate_value" in token["pre_types"][step]:
        try:
            cand = list(range(word_info[f"{idx2word[str(ptok_nos[-2])]}#column values"]["start_id"],
                              word_info[f"{idx2word[str(ptok_nos[-2])]}#column values"]["end_id"] + 1))
        except:
            print("#value/#aggregate_value")
    # 1.2) special values: count()/avg()/sum() numeric aggregate values.
    # only one.
    elif idx2word[str(ptok_nos[-3])] in constants.aggregator[2:] and \
            "#numeric_aggregate_value" in token["pre_types"][step]:
        cand = [constants.UNK]  # UNK
    # more than one.
    elif "#numeric_aggregate_value" in token["pre_types"][step]:
        try:
            cand = list(range(word_info[f"{idx2word[str(ptok_nos[-2])]}#column values"]["start_id"],
                              word_info[f"{idx2word[str(ptok_nos[-2])]}#column values"]["end_id"] + 1))
        except:
            print("#numeric_aggregate_value")

    # todo: perturbation step constraint, already decoded words difference (not forcibly).
    #  group by / order by / having clause.
    if np.sum(np.array(token["pno_tokens"][:step]) != np.array(ptok_nos)) >= max_diff \
            and token["pno_tokens"][step] in cand:
        return [token["pno_tokens"][step]]
    else:
        return cand


def sql2vec(sql_tokens, word2idx):
    """
    format the sql in token_res already.
    sql = mosqlparse.format(mosqlparse.parse(sql))
    :param sql_tokens:
    :param word2idx:
    :return:
    """
    vectors = list()
    for item in sql_tokens:

        # vec = list(map(lambda x: word2idx.get(
        #     str(x).strip("'").split(".")[-1], 3),
        #                item["pre_tokens"]))
        vec = list()
        for i in range(len(item["pre_tokens"])):
            key = str(item['pre_tokens'][i]).strip("'").strip(" ")
            if "operator" in item["pre_types"][i] and \
                    item["pre_tokens"][i] == "<>":
                key = "!="
            elif "aggregator" in item["pre_types"][i]:
                key = key.lower()
            elif "column" in item["pre_types"][i]:
                # todo: `table`.`column`
                key = key.split(".")[-1]
            # todo: newly added for `<unk>`.
            elif "#numeric_aggregate_value" in item["pre_types"][i]:
                key = "<unk>"
            elif "value" in item["pre_types"][i]:
                key = f"{item['pre_types'][i].split('#')[0]}#_#{key}"

            # if word2idx.get(key, 1) == 1 and \
            #         "numeric_aggregate_value" not in item['pre_types'][i]:
            #     print(item['pre_types'][i])
            vec.append(word2idx.get(key, 1))  # <unk>: 1
        # if np.sum(np.array(vec) == 1) != 0:
        #     print(1)
        vectors.append(vec)

    return vectors


def vec2sql(sql_tokens, sql_vectors, idx2word, col_info, mode="without_table"):
    """

    :param sql_tokens: "<unk>"
    :param sql_vectors:
    :param idx2word:
    :param col_info:
    :param mode:
    :return:
    """

    columns = list(col_info.keys())
    tables = list(set([col_info[key]["table"] for key in col_info]))

    sql_res = list()
    for tok, vec in zip(sql_tokens, sql_vectors):
        res = {"sql_text": "", "sql_token": list(), "pno_tokens": list(map(int, vec))}
        for to, no in zip(tok["pre_tokens"], vec):
            # Filter the special token.
            if no in [constants.BOS, constants.EOS, constants.PAD]:
                continue

            pre1, pre2, pre3 = "", "", ""
            if len(res["sql_token"]) - 1 > 0:
                pre1 = res["sql_token"][len(res["sql_token"]) - 1]
            if len(res["sql_token"]) - 2 > 0:
                pre2 = res["sql_token"][len(res["sql_token"]) - 2]
            if len(res["sql_token"]) - 3 > 0:
                pre3 = res["sql_token"][len(res["sql_token"]) - 3]

            cur = idx2word[str(no)]
            # for numeric aggregate value
            if cur == "<unk>":
                cur = to
            res["sql_token"].append(cur)

            if pre1 in constants.aggregator and cur in columns:
                if mode == "with_table":
                    cur = f"({col_info[cur]['table']}.{cur})"
                else:
                    cur = f"({cur})"
            # todo: constants.operator[:2] -> constants.aggregator[:2]
            elif pre3 in constants.aggregator[:2] and \
                    pre2 in columns and \
                    cur in col_info[pre2]["value"] and \
                    col_info[pre2]["type"] not in ["integer", "numeric"]:
                cur = f"'{cur}'"
            # todo: constants.operator -> constants.aggregator
            elif pre3 not in constants.aggregator and \
                    pre2 in columns and \
                    cur in col_info[pre2]["value"] and \
                    col_info[pre2]["type"] not in ["integer", "numeric"]:
                cur = f"'{cur}'"

            if (pre1 in columns or pre1 in tables) and \
                    (cur in columns or cur in tables):
                if cur in columns and mode == "with_table":
                    res["sql_text"] += f", {col_info[cur]['table']}.{cur}"
                else:
                    res["sql_text"] += f", {cur}"
            elif pre1 in columns and cur in constants.aggregator:
                res["sql_text"] += f", {cur}"
            elif pre1 in constants.order_by_key and \
                    (cur in columns or cur in constants.aggregator):
                if cur in columns and mode == "with_table":
                    res["sql_text"] += f", {col_info[cur]['table']}.{cur}"
                else:
                    res["sql_text"] += f", {cur}"
            elif pre1 in constants.aggregator:
                res["sql_text"] += cur
            else:
                if cur in columns and mode == "with_table":
                    res["sql_text"] += f" {col_info[cur]['table']}.{cur}"
                else:
                    res["sql_text"] += f" {cur}"

        res["sql_text"] = res["sql_text"].strip(" ")
        sql_res.append(res)

    return sql_res


def random_gen(sql_token, word2idx, idx2word, word_info,
               col_info, mode="value", max_diff=5, perturb_prop=0.5, seed=666):
    """

    :param sql_token:
    :param word2idx:
    :param idx2word:
    :param word_info:
    :param col_info:
    :param mode:
    :param max_diff:
    :param seed:
    :return:
    """
    random.seed(seed)
    valid_tokens, except_tokens, sql_vectors = list(), list(), list()
    for ino, token in tqdm(enumerate(sql_token)):
        # if ino == 616:
        #     print(1)
        try:
            vec = token["pno_tokens"]
            table = [token["pre_tokens"][i] for i, typ in
                     enumerate(token["pre_types"]) if "table" in typ]
            column_left = [token["pno_tokens"][i] for i, typ in
                           enumerate(token["pre_types"]) if
                           typ == "select#column" or
                           (typ == "select#aggregate_column" and
                            idx2word[str(token["pno_tokens"][i - 1])] in constants.aggregator[:2]) or
                           typ == "where#column"]
            sql_tok = list()
            for step in range(len(token["pre_types"])):
                if mode == "all":
                    cand = valid_cand(token, table, step, sql_tok, word2idx,
                                      idx2word, word_info, col_info, max_diff=max_diff)
                elif mode == "column":
                    cand = valid_cand_col(token, table, step, sql_tok, column_left, word2idx,
                                          idx2word, word_info, col_info, max_diff=max_diff)
                elif mode == "value":
                    cand = valid_cand_val(token, table, step, sql_tok, word2idx,
                                          idx2word, word_info, col_info, max_diff=max_diff)

                if random.uniform(0, 1) > perturb_prop and vec[step] in cand:
                    selected = vec[step]
                    sql_tok.append(selected)
                    if selected in column_left and \
                            (token["pre_types"][step] == "select#column" or
                             (token["pre_types"][step] == "select#aggregate_column" and
                              idx2word[str(token["pno_tokens"][step - 1])] in constants.aggregator[:2]) or
                             token["pre_types"][step] == "where#column"):
                        column_left.remove(selected)
                else:
                    selected = random.choice(cand)
                    sql_tok.append(selected)
                    if selected in column_left and \
                            (token["pre_types"][step] == "select#column" or
                             (token["pre_types"][step] == "select#aggregate_column" and
                              idx2word[str(token["pno_tokens"][step - 1])] in constants.aggregator[:2]) or
                             token["pre_types"][step] == "where#column"):
                        column_left.remove(selected)
                    # try:
                    #     sql_tok.append(random.choice(cand))
                    # except:
                    #     print("sql_tok.append(random.choice(cand))")
                # sql_tok.append(random.choice(cand))
            valid_tokens.append(token)
            sql_vectors.append(sql_tok)
        except:
            traceback.print_exc()
            cand = valid_cand(token, table, step, sql_tok, word2idx,
                              idx2word, word_info, col_info, max_diff=max_diff)
            # cand = valid_cand_col(token, table, step, sql_tok, column_left, word2idx,
            #                       idx2word, word_info, col_info, max_diff=max_diff)
            except_tokens.append((ino, token))

    return valid_tokens, except_tokens, sql_vectors


if __name__ == "__main__":
    # sql_token_json = "/data/wz/index/attack/visual_" \
    #                  "rewrite/data/small_sample.json"

    sql_token_json = "/data/wz/index/attack/visual_rewrite/data/" \
                     "val_swap_cost_value_filter_format_tpch_queries_all.json"

    # sql_cand = get_candidate(sql_token_json)
    # sql_vec = random_gen(sql_cand)
    sql_token_json = "/data/wz/index/attack/resource/visrew_queries/" \
                     "perturb_rcost_value_filter_format_all.json"

    # 666 -> p_name (repeated)
    sql_token_json = "/data/wz/index/attack/resource/visrew_queries/" \
                     "valswap_index_perturb_rcost_value_filter_format_all.json"
    with open(sql_token_json, "r") as rf:
        sql_token = json.load(rf)

    valid_token, except_token, sql_vec = random_gen(sql_token, max_diff=5)
    sql_res = vec2sql(valid_token, sql_vec)

    # print(sql_res[3666]['sql_text'] + ";" + pre_data.vec2sql([valid_token[3666]], sql2vec([valid_token[3666]]))[0]['sql_text'] + ";")
    print(1)

    with open("src.sql", "w") as wf:
        for token in valid_token:
            wf.writelines(token["sql"] + "\n")

    with open("tgt.sql", "w") as wf:
        for sql in sql_res:
            wf.writelines(sql["sql_text"] + "\n")
