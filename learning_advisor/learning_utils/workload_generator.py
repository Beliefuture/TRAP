import copy
import logging
import random

import numpy as np

from learning_advisor.learning_utils.cost_evaluation import CostEvaluation
import learning_advisor.learning_utils.embedding_utils as embedding_utils
from learning_advisor.learning_utils.swirl_com import get_utilized_indexes

from heuristic_advisor.heuristic_utils.candidate_generation import (
    candidates_per_query,
    syntactically_relevant_indexes,
)
from heuristic_advisor.heuristic_utils.postgres_dbms import PostgresDatabaseConnector
from learning_advisor.learning_utils.workload import Query, Workload

from .workload_embedder import WorkloadEmbedder


class WorkloadGenerator(object):
    def __init__(self, work_config, work_type, work_file,
                 db_config, schema_columns, random_seed, experiment_id=None,
                 is_filter_workload_cols=True, is_filter_utilized_cols=False):
        assert work_config["benchmark"] in [
            "TPCH",
            "TPCDS",
            "JOB",
        ], f"Benchmark '{work_config['benchmark']}' is currently not supported."

        self.rnd = random.Random()
        self.rnd.seed(random_seed)
        self.np_rnd = np.random.default_rng(seed=random_seed)

        # For create view statement differentiation
        self.experiment_id = experiment_id

        self.db_config = db_config
        # all the columns in the schema after the filter of `TableNumRowsFilter`.
        self.schema_columns = schema_columns

        self.benchmark = work_config["benchmark"]  # default: TPC-H

        self.is_varying_frequencies = work_config["varying_frequencies"]  # default: false
        if work_type == "template":
            self.number_of_query_classes = self._set_number_of_query_classes()
            # default: 2, 17, 20 for TPC-H
            self.excluded_query_classes = set(work_config["excluded_query_classes"])
            self.query_texts = self._retrieve_query_texts(work_file)
        else:
            self.query_texts = self._load_no_temp_workload(work_file)
            self.number_of_query_classes = len(self.query_texts)
            self.excluded_query_classes = set()

        self.query_classes = set(range(1, self.number_of_query_classes + 1))
        self.available_query_classes = self.query_classes - self.excluded_query_classes

        self.globally_indexable_columns = self._select_indexable_columns(is_filter_workload_cols,
                                                                         is_filter_utilized_cols)
        assert work_config["size"] > 1 or (work_config["training_instances"] + work_config["validation_testing"]["number_of_workloads"]
               <= self.number_of_query_classes and work_config["size"] == 1), "Can not generate the workload satisfied!"

        num_validation_instances = work_config["validation_testing"]["number_of_workloads"]
        num_test_instances = work_config["validation_testing"]["number_of_workloads"]
        self.wl_validation = []
        self.wl_testing = []

        if work_config["similar_workloads"] and work_config["unknown_queries"] == 0:
            assert self.is_varying_frequencies, "Similar workloads can only be created with varying frequencies."
            self.wl_validation = [None]
            self.wl_testing = [None]
            _, self.wl_validation[0], self.wl_testing[0] = self._generate_workloads(
                0, num_validation_instances, num_test_instances, work_config["size"])
            if "query_class_change_frequency" not in work_config \
                    or work_config["query_class_change_frequency"] is None:
                self.wl_training = self._generate_similar_workloads(work_config["training_instances"], work_config["size"])
            else:
                self.wl_training = self._generate_similar_workloads_qccf(
                    work_config["training_instances"], work_config["size"], work_config["query_class_change_frequency"])
        elif work_config["unknown_queries"] > 0:
            assert (
                    work_config["validation_testing"]["unknown_query_probabilities"][-1] > 0
            ), "Query unknown_query_probabilities should be larger 0."

            embedder_connector = PostgresDatabaseConnector(self.db_config, autocommit=True)
            embedder = WorkloadEmbedder(
                query_texts=self.query_texts,
                representation_size=0,
                database_connector=embedder_connector,
                globally_index_candidates=[list(map(lambda x: [x], self.globally_indexable_columns))],
                retrieve_plans=True)
            self.unknown_query_classes = embedding_utils.which_queries_to_remove(
                embedder.plans, work_config["unknown_queries"], random_seed)  # self.excluded_query_classes
            self.unknown_query_classes = frozenset(self.unknown_query_classes) - self.excluded_query_classes

            # `missing_classes`: caused by the operation of `excluded`.
            missing_classes = work_config["unknown_queries"] - len(self.unknown_query_classes)
            # complement if missing, randomly sampled from the set of available query class.
            self.unknown_query_classes = self.unknown_query_classes | frozenset(
                self.rnd.sample(self.available_query_classes - frozenset(self.unknown_query_classes), missing_classes)
            )
            assert len(self.unknown_query_classes) == work_config["unknown_queries"]
            embedder_connector.close()

            self.known_query_classes = self.available_query_classes - frozenset(self.unknown_query_classes)
            embedder = None

            for query_class in self.excluded_query_classes:
                assert query_class not in self.unknown_query_classes

            logging.critical(f"Global unknown query classes: {sorted(self.unknown_query_classes)}")
            logging.critical(f"Global known query classes: {sorted(self.known_query_classes)}")

            for unknown_query_probability in work_config["validation_testing"]["unknown_query_probabilities"]:
                _, wl_validation, wl_testing = self._generate_workloads(
                    0,
                    num_validation_instances,
                    num_test_instances,
                    work_config["size"],
                    unknown_query_probability=unknown_query_probability,
                )
                self.wl_validation.append(wl_validation)
                self.wl_testing.append(wl_testing)

            assert (
                    len(self.wl_validation)
                    == len(work_config["validation_testing"]["unknown_query_probabilities"])
                    == len(self.wl_testing)
            ), "Validation/Testing workloads length fail"

            # We are temporarily restricting the available query classes now to exclude certain classes for training
            original_available_query_classes = self.available_query_classes
            self.available_query_classes = self.known_query_classes

            if work_config["similar_workloads"]:
                if work_config["query_class_change_frequency"] is not None:
                    logging.critical(
                        f"Similar workloads with query_class_change_frequency: {work_config['query_class_change_frequency']}"
                    )
                    self.wl_training = self._generate_similar_workloads_qccf(
                        work_config["training_instances"], work_config["size"], work_config["query_class_change_frequency"]
                    )
                else:
                    self.wl_training = self._generate_similar_workloads(work_config["training_instances"], work_config["size"])
            else:
                self.wl_training, _, _ = self._generate_workloads(work_config["training_instances"], 0, 0, work_config["size"])
            self.available_query_classes = original_available_query_classes
        else:
            self.wl_validation = [None]
            self.wl_testing = [None]
            self.wl_training, self.wl_validation[0], self.wl_testing[0] = self._generate_workloads(
                work_config["training_instances"], num_validation_instances, num_test_instances, work_config["size"])

        logging.critical(f"Sample instances from training workloads: {self.rnd.sample(self.wl_training, 10)}")
        logging.info("Finished generating workloads.")

    def _set_number_of_query_classes(self):
        if self.benchmark == "TPCH":
            return 22
        elif self.benchmark == "TPCDS":
            return 99
        elif self.benchmark == "JOB":
            return 113
        else:
            raise ValueError("Unsupported Benchmark type provided, only TPCH, TPCDS, and JOB supported.")

    def _retrieve_query_texts(self, work_file):
        query_files = [open(f"{work_file}/{self.benchmark}/{self.benchmark}_{file_number}.txt", "r")
                       for file_number in range(1, self.number_of_query_classes + 1)]

        finished_queries = []
        for query_file in query_files:
            queries = query_file.readlines()[:1]
            # remove `limit x`, replace the name of the view with the experiment id
            queries = self._preprocess_queries(queries)

            finished_queries.append(queries)
            query_file.close()

        assert len(finished_queries) == self.number_of_query_classes

        return finished_queries

    def _load_no_temp_workload(self, work_file):
        with open(work_file, "r") as rf:
            sql_list = rf.readlines()

        finished_queries = []
        for sql in sql_list:
            # remove `limit x`, replace the name of the view with the experiment id
            queries = self._preprocess_queries([sql])
            finished_queries.append(queries)

        logging.info(f"Load the workload from `{work_file}`.")

        return finished_queries

    def _preprocess_queries(self, queries):
        processed_queries = []
        for query in queries:
            query = query.replace("limit 100", "")
            query = query.replace("limit 20", "")
            query = query.replace("limit 10", "")
            query = query.strip()

            if "create view revenue0" in query:
                query = query.replace("revenue0", f"revenue0_{self.experiment_id}")

            processed_queries.append(query)

        return processed_queries

    def _select_indexable_columns(self, is_filter_workload_cols, is_filter_utilized_cols):
        if is_filter_workload_cols:
            available_query_classes = tuple(self.available_query_classes)
            query_class_frequencies = tuple([1 for _ in range(len(available_query_classes))])

            logging.info(f"Selecting indexable columns on {len(available_query_classes)} query classes.")

            # load the workload for later indexable columns, choose one query per query class randomly.
            workload = self._workloads_from_tuples([(available_query_classes, query_class_frequencies)])[0]

            # return the sorted(by default) list of the indexable columns given the workload.
            indexable_columns = workload.indexable_columns(return_sorted=True)

            if is_filter_utilized_cols:
                indexable_columns = self._only_utilized_indexes(indexable_columns)
        else:
            indexable_columns = self.schema_columns

        selected_columns = []
        global_column_id = 0
        for column in self.schema_columns:
            if column in indexable_columns:
                column.global_column_id = global_column_id
                global_column_id += 1

                selected_columns.append(column)

        return selected_columns

    def _workloads_from_tuples(self, tuples, unknown_query_probability=None):
        workloads = []
        unknown_query_probability = "" if unknown_query_probability is None else unknown_query_probability

        for tupl in tuples:
            query_classes, query_class_frequencies = tupl

            # single workload: len(query_classes/query_class_frequencies) = number of queries
            queries = []  # select one query from one query_class
            for query_class, frequency in zip(query_classes, query_class_frequencies):
                # self.query_texts is list of lists.
                # Outer list for query classes, inner list for instances of this class.
                query_text = self.rnd.choice(self.query_texts[query_class - 1])
                query = Query(query_class, query_text, frequency=frequency)

                # retrieve the indexable columns(in the WHERE clause) given the query.
                self._store_indexable_columns(query)
                assert len(query.columns) > 0, f"Query columns should have length > 0: {query.text}"

                queries.append(query)

            assert isinstance(queries, list), f"Queries is not of type list but of {type(queries)}"

            previously_unseen_queries = (round(unknown_query_probability * len(queries))
                                         if unknown_query_probability != "" else 0)
            workloads.append(Workload(queries,
                                      description=f"Contains {previously_unseen_queries} previously unseen queries."))

        return workloads

    def _store_indexable_columns(self, query):
        if self.benchmark != "JOB":
            for column in self.schema_columns:
                if column.name in query.text.lower():
                    query.columns.append(column)
        else:
            query_text = query.text.lower()
            assert "WHERE" in query_text, f"Query without WHERE clause encountered: {query_text} in {query.nr}"

            split = query_text.split("WHERE")
            assert len(split) == 2, "Query split for JOB query contains subquery"
            query_text_before_where = split[0]
            query_text_after_where = split[1]

            for column in self.schema_columns:
                if column.name in query_text_after_where and f"{column.table.name} " in query_text_before_where:
                    query.columns.append(column)

    def _only_utilized_indexes(self, indexable_columns):
        frequencies = [1 for _ in range(len(self.available_query_classes))]
        workload_tuple = (self.available_query_classes, frequencies)
        workload = self._workloads_from_tuples([workload_tuple])[0]

        candidates = candidates_per_query(workload,
                                          max_index_width=1,
                                          candidate_generator=syntactically_relevant_indexes)

        connector = PostgresDatabaseConnector(self.db_config, autocommit=True)
        connector.drop_indexes()
        cost_evaluation = CostEvaluation(connector)

        utilized_indexes, query_details = get_utilized_indexes(workload, candidates, cost_evaluation, True)

        columns_of_utilized_indexes = set()
        for utilized_index in utilized_indexes:
            column = utilized_index.columns[0]
            columns_of_utilized_indexes.add(column)

        output_columns = columns_of_utilized_indexes & set(indexable_columns)
        excluded_columns = set(indexable_columns) - output_columns
        logging.critical(f"Excluding columns based on utilization:\n   {excluded_columns}")

        return output_columns

    def _generate_workloads(self, train_instances, validation_instances,
                            test_instances, size, unknown_query_probability=None):
        required_unique_workloads = train_instances + validation_instances + test_instances

        unique_workload_tuples = set()
        while required_unique_workloads > len(unique_workload_tuples):
            workload_tuple = self._generate_random_workload(size, unknown_query_probability)
            unique_workload_tuples.add(workload_tuple)

        validation_tuples = self.rnd.sample(unique_workload_tuples, validation_instances)
        unique_workload_tuples = unique_workload_tuples - set(validation_tuples)

        test_workload_tuples = self.rnd.sample(unique_workload_tuples, test_instances)
        unique_workload_tuples = unique_workload_tuples - set(test_workload_tuples)

        assert len(unique_workload_tuples) == train_instances
        train_workload_tuples = unique_workload_tuples

        assert (len(train_workload_tuples) + len(test_workload_tuples)
                + len(validation_tuples) == required_unique_workloads)

        # list(Object(Workload))
        validation_workloads = self._workloads_from_tuples(validation_tuples, unknown_query_probability)
        test_workloads = self._workloads_from_tuples(test_workload_tuples, unknown_query_probability)
        train_workloads = self._workloads_from_tuples(train_workload_tuples, unknown_query_probability)

        return train_workloads, validation_workloads, test_workloads

    def _generate_random_workload(self, size, unknown_query_probability=None):
        assert size <= self.number_of_query_classes, "Cannot generate workload with more queries than query classes"

        # 1) determine query class
        if unknown_query_probability is not None:
            number_of_unknown_queries = round(size * unknown_query_probability)  # default 0 digits
            number_of_known_queries = size - number_of_unknown_queries
            assert number_of_known_queries + number_of_unknown_queries == size

            known_query_classes = self.rnd.sample(self.known_query_classes, number_of_known_queries)
            unknown_query_classes = self.rnd.sample(self.unknown_query_classes, number_of_unknown_queries)
            query_classes = known_query_classes
            query_classes.extend(unknown_query_classes)
            workload_query_classes = tuple(query_classes)
            assert len(workload_query_classes) == size
        else:
            workload_query_classes = tuple(self.rnd.sample(self.available_query_classes, size))

        # 2) determine query frequencies
        if self.is_varying_frequencies:
            query_class_frequencies = tuple(list(self.np_rnd.integers(1, 10000, size)))
        else:
            query_class_frequencies = tuple([1 for _ in range(size)])

        workload_tuple = (workload_query_classes, query_class_frequencies)

        return workload_tuple

    def _generate_similar_workloads(self, instances, size):
        assert size <= len(self.available_query_classes), \
            "Cannot generate workload with more queries than query classes"

        workload_tuples = []
        query_classes = self.rnd.sample(self.available_query_classes, size)
        available_query_classes = self.available_query_classes - frozenset(query_classes)
        frequencies = list(self.np_rnd.zipf(1.5, size))

        workload_tuples.append((copy.copy(query_classes), copy.copy(frequencies)))

        for workload_idx in range(instances - 1):
            # Remove a random element
            idx_to_remove = self.rnd.randrange(len(query_classes))
            query_classes.pop(idx_to_remove)
            frequencies.pop(idx_to_remove)

            # Draw a new random element, the removed one is excluded
            query_classes.append(self.rnd.sample(available_query_classes, 1)[0])
            frequencies.append(self.np_rnd.zipf(1.5, 1)[0])

            frequencies[self.rnd.randrange(len(query_classes))] = self.np_rnd.zipf(1.5, 1)[0]

            available_query_classes = self.available_query_classes - frozenset(query_classes)
            workload_tuples.append((copy.copy(query_classes), copy.copy(frequencies)))

        workloads = self._workloads_from_tuples(workload_tuples)

        return workloads

    # This version uses the same query id selection for `query_class_change_frequency` workloads.
    def _generate_similar_workloads_qccf(self, instances, size, query_class_change_frequency):
        assert size <= len(
            self.available_query_classes
        ), "Cannot generate workload with more queries than query classes"

        workload_tuples = []

        while len(workload_tuples) < instances:
            if len(workload_tuples) % query_class_change_frequency == 0:
                query_classes = self.rnd.sample(self.available_query_classes, size)

            frequencies = list(self.np_rnd.integers(1, 10000, size))
            workload_tuples.append((copy.copy(query_classes), copy.copy(frequencies)))

        workloads = self._workloads_from_tuples(workload_tuples)

        return workloads
