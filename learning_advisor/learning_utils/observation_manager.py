import logging

import numpy as np
from gym import spaces

VERY_HIGH_BUDGET = 100_000_000_000


class ObservationManager(object):
    def __init__(self, number_of_columns):
        self.number_of_columns = number_of_columns

    def _init_episode(self, state_fix_for_episode):
        self.episode_budget = state_fix_for_episode["budget"]
        if self.episode_budget is None:
            self.episode_budget = VERY_HIGH_BUDGET

        self.initial_cost = state_fix_for_episode["initial_cost"]

    def init_episode(self, state_fix_for_episode):
        raise NotImplementedError

    def get_observation(self, environment_state):
        raise NotImplementedError

    def get_observation_space(self):
        observation_space = spaces.Box(
            low=self._create_low_boundaries(), high=self._create_high_boundaries(), shape=self._create_shape()
        )

        logging.info(f"Creating ObservationSpace with {self.number_of_features} features.")

        return observation_space

    def _create_shape(self):
        return (self.number_of_features,)

    def _create_low_boundaries(self):
        low = [-np.inf for feature in range(self.number_of_features)]

        return np.array(low)

    def _create_high_boundaries(self):
        high = [np.inf for feature in range(self.number_of_features)]

        return np.array(high)


class EmbeddingObservationManager(ObservationManager):
    def __init__(self, number_of_columns, config):
        ObservationManager.__init__(self, number_of_columns)

        self.workload_embedder = config["workload_embedder"]
        self.representation_size = self.workload_embedder.representation_size
        self.workload_size = config["workload_size"]

        self.number_of_features = (
                self.number_of_columns
                + (
                        self.representation_size * self.workload_size
                )
                + self.workload_size
                + 1
                + 1
                + 1
                + 1
        )

    def _init_episode(self, state_fix_for_episode):
        episode_workload = state_fix_for_episode["workload"]
        self.frequencies = np.array(EmbeddingObservationManager._get_frequencies_from_workload(episode_workload))

        super()._init_episode(state_fix_for_episode)

    def init_episode(self, state_fix_for_episode):
        raise NotImplementedError

    def get_observation(self, environment_state):
        if self.UPDATE_EMBEDDING_PER_OBSERVATION:
            workload_embedding = np.array(self.workload_embedder.get_embeddings(environment_state["plans_per_query"]))
        else:
            if self.workload_embedding is None:
                self.workload_embedding = np.array(
                    self.workload_embedder.get_embeddings(environment_state["plans_per_query"])
                )

            workload_embedding = self.workload_embedding

        observation = np.array(environment_state["action_status"])
        observation = np.append(observation, workload_embedding)
        observation = np.append(observation, self.frequencies)
        observation = np.append(observation, self.episode_budget)
        observation = np.append(observation, environment_state["current_storage_consumption"])
        observation = np.append(observation, self.initial_cost)
        observation = np.append(observation, environment_state["current_cost"])

        return observation

    @staticmethod
    def _get_frequencies_from_workload(workload):
        frequencies = []
        for query in workload.queries:
            frequencies.append(query.frequency)
        return frequencies


# All managers are capable of handling single and multi-attribute indexes now.
class SingleColumnIndexWorkloadEmbeddingObservationManager(EmbeddingObservationManager):
    def __init__(self, number_of_columns, config):
        super().__init__(number_of_columns, config)

        self.UPDATE_EMBEDDING_PER_OBSERVATION = False

    def init_episode(self, state_fix_for_episode):
        super()._init_episode(state_fix_for_episode)

        self.workload_embedding = np.array(self.workload_embedder.get_embeddings(state_fix_for_episode["workload"]))


# All managers are capable of handling single and multi-attribute indexes now.
class SingleColumnIndexPlanEmbeddingObservationManager(EmbeddingObservationManager):
    def __init__(self, number_of_columns, config):
        super().__init__(number_of_columns, config)

        self.UPDATE_EMBEDDING_PER_OBSERVATION = True

    def init_episode(self, state_fix_for_episode):
        super()._init_episode(state_fix_for_episode)


# All managers are capable of handling single and multi-attribute indexes now.
class SingleColumnIndexPlanEmbeddingObservationManagerWithoutPlanUpdates(EmbeddingObservationManager):
    def __init__(self, number_of_columns, config):
        super().__init__(number_of_columns, config)

        self.UPDATE_EMBEDDING_PER_OBSERVATION = False

    def init_episode(self, state_fix_for_episode):
        super()._init_episode(state_fix_for_episode)

        self.workload_embedding = None


# All managers are capable of handling single and multi-attribute indexes now.
class SingleColumnIndexPlanEmbeddingObservationManagerWithCost(EmbeddingObservationManager):
    def __init__(self, number_of_columns, config):
        super().__init__(number_of_columns, config)

        self.UPDATE_EMBEDDING_PER_OBSERVATION = True

        # This overwrites EmbeddingObservationManager's features
        self.number_of_features = (
            self.number_of_columns
            + (self.representation_size * self.workload_size)
            + self.workload_size
            + self.workload_size
            + 1
            + 1
            + 1
            + 1
        )

    def init_episode(self, state_fix_for_episode):
        super()._init_episode(state_fix_for_episode)

    # This overwrite EmbeddingObservationManager.get_observation() because further features are added
    def get_observation(self, environment_state):
        workload_embedding = np.array(self.workload_embedder.get_embeddings(environment_state["plans_per_query"]))
        observation = np.array(environment_state["action_status"])  # 54
        observation = np.append(observation, workload_embedding)  # 50
        observation = np.append(observation, environment_state["costs_per_query"])  # 1
        observation = np.append(observation, self.frequencies)  # 1
        observation = np.append(observation, self.episode_budget)  # 1
        observation = np.append(observation, environment_state["current_storage_consumption"])  # 1
        observation = np.append(observation, self.initial_cost)  # 1
        observation = np.append(observation, environment_state["current_cost"])  # 1

        return observation


class SingleColumnIndexColumnObservationManagerWithCost(ObservationManager):
    def __init__(self, number_of_columns, config):
        super().__init__(number_of_columns)

        self.UPDATE_EMBEDDING_PER_OBSERVATION = True
        self.number_of_query_classes = config["workload_size"]
        self.workload_size = config["workload_size"]

        # This overwrites EmbeddingObservationManager's features
        self.number_of_features = (
            self.number_of_columns
            + (self.number_of_query_classes * self.number_of_columns)
            + self.workload_size
            + self.workload_size
            + 1
            + 1
            + 1
            + 1
        )

        self._workload_matrix = None

    def _update_episode_fix_data(self, state_fix_for_episode):
        self._workload_matrix = [
            [0 for m in range(self.number_of_columns)] for k in range(self.number_of_query_classes)
        ]
        for query_id, query in enumerate(state_fix_for_episode["workload"].queries):
            # Account for zero indexing
            # query_id = query.nr - 1
            for column in query.columns:
                if column.global_column_id is None:
                    continue
                self._workload_matrix[query_id][column.global_column_id] = 1

    def init_episode(self, state_fix_for_episode):
        episode_workload = state_fix_for_episode["workload"]
        self.frequencies = np.array(EmbeddingObservationManager._get_frequencies_from_workload(episode_workload))

        super()._init_episode(state_fix_for_episode)
        self._update_episode_fix_data(state_fix_for_episode)

    # This overwrite EmbeddingObservationManager.get_observation() because further features are added
    def get_observation(self, environment_state):
        observation = np.array(environment_state["action_status"])  # 54
        observation = np.append(observation, self._workload_matrix)
        observation = np.append(observation, environment_state["costs_per_query"])  # 1
        observation = np.append(observation, self.frequencies)  # 1
        observation = np.append(observation, self.episode_budget)  # 1
        observation = np.append(observation, environment_state["current_storage_consumption"])  # 1
        observation = np.append(observation, self.initial_cost)  # 1
        observation = np.append(observation, environment_state["current_cost"])  # 1

        return observation

    @staticmethod
    def _get_frequencies_from_workload(workload):
        frequencies = []
        for query in workload.queries:
            frequencies.append(query.frequency)
        return frequencies


class SingleColumnIndexObservationManager(ObservationManager):
    def __init__(self, number_of_columns, config):
        ObservationManager.__init__(self, number_of_columns)

        self.number_of_query_classes = config["number_of_query_classes"]

        self.number_of_features = (
                self.number_of_columns
                + self.number_of_query_classes
                + 1
                + 1
                + 1
                + 1
        )

    def init_episode(self, state_fix_for_episode):
        episode_workload = state_fix_for_episode["workload"]
        super()._init_episode(state_fix_for_episode)
        self.frequencies = np.array(self._get_frequencies_from_workload_wide(episode_workload))

    def get_observation(self, environment_state):
        observation = np.array(environment_state["action_status"])
        observation = np.append(observation, self.frequencies)
        observation = np.append(observation, self.episode_budget)
        observation = np.append(observation, environment_state["current_storage_consumption"])
        observation = np.append(observation, self.initial_cost)
        observation = np.append(observation, environment_state["current_cost"])

        return observation

    def _get_frequencies_from_workload_wide(self, workload):
        frequencies = [0 for query in range(self.number_of_query_classes)]

        for query in workload.queries:
            # query numbers stat at 1
            frequencies[query.nr - 1] = query.frequency

        return frequencies


class UnknownQueriesObservationManager(ObservationManager):
    def __init__(self, number_of_columns, config):
        ObservationManager.__init__(self, number_of_columns)

        self.number_of_query_classes = config["number_of_query_classes"]

        self.number_of_features = (
                self.number_of_columns
                + 1
                + 1
                + 1
                + 1
        )

    def init_episode(self, state_fix_for_episode):
        super()._init_episode(state_fix_for_episode)

    def get_observation(self, environment_state):
        observation = np.array(environment_state["action_status"])
        observation = np.append(observation, self.episode_budget)
        observation = np.append(observation, environment_state["current_storage_consumption"])
        observation = np.append(observation, self.initial_cost)
        observation = np.append(observation, environment_state["current_cost"])

        return observation


class DRLindaObservationManager(ObservationManager):
    def __init__(self, number_of_columns, config):
        ObservationManager.__init__(self, number_of_columns)
        self.number_of_query_classes = config["workload_size"]

        self.number_of_features = (
                self.number_of_columns
                + (self.number_of_query_classes * self.number_of_columns)
                + self.number_of_columns
        )

        self._workload_matrix = None
        self._access_vector = None

    def _update_episode_fix_data(self, state_fix_for_episode):
        self._workload_matrix = [
            [0 for m in range(self.number_of_columns)] for k in range(self.number_of_query_classes)
        ]
        self._access_vector = [0 for m in range(self.number_of_columns)]

        for query_id, query in enumerate(state_fix_for_episode["workload"].queries):
            # Account for zero indexing
            # query_id = query.nr - 1
            for column in query.columns:
                if column.global_column_id is None:
                    continue
                self._workload_matrix[query_id][column.global_column_id] = 1
                self._access_vector[column.global_column_id] += query.frequency

    def init_episode(self, state_fix_for_episode):
        super()._init_episode(state_fix_for_episode)

        self._update_episode_fix_data(state_fix_for_episode)

    def get_observation(self, environment_state):
        assert self._workload_matrix is not None
        assert self._access_vector is not None

        observation = np.array(environment_state["action_status"])
        observation = np.append(observation, self._workload_matrix)
        observation = np.append(observation, self._access_vector)

        return observation
