import os
import copy
import importlib
import logging
import pickle
import numpy as np

import sys
sys.path.append("..")

from learning_advisor.learning_infer import get_eval_env
from learning_advisor.experiment import Experiment
from learning_advisor.gym_db.common import EnvironmentType
from learning_advisor.learning_utils import swirl_com
from learning_advisor.learning_utils.configuration_parser import ConfigurationParser
from learning_advisor.learning_utils.swirl_com import set_logger
from learning_advisor.learning_utils.workload_generator import WorkloadGenerator
from learning_advisor.learning_utils.workload import Query, Workload


if __name__ == "__main__":
    parser = swirl_com.get_parser()
    args = parser.parse_args()

    logging.info(f"The training mode is `{args.train_mode}`.")

    # load the configuration, create the experiment folder.
    experiment = Experiment(args)
    logging.info(f"The value of `parallel_environments` is `{experiment.exp_config['parallel_environments']}`.")

    # only stable_baselines2 supported.
    if experiment.exp_config["rl_algorithm"]["stable_baselines_version"] == 2:
        from stable_baselines.common.callbacks import EvalCallbackWithTBRunningAverage
        from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

        # <class 'stable_baselines.ppo2.ppo2.PPO2'>
        algorithm_class = getattr(
            importlib.import_module("learning_advisor.stable_baselines"),
            experiment.exp_config["rl_algorithm"]["algorithm"])
    else:
        raise ValueError

    experiment.prepare()

    ParallelEnv = SubprocVecEnv if experiment.exp_config["parallel_environments"] > 1 else DummyVecEnv
    training_env = ParallelEnv([experiment.make_env(env_id,
                                                    environment_type=EnvironmentType.TRAINING,
                                                    workloads_in=None,
                                                    db_config=experiment.schema.db_config)
                                for env_id in range(experiment.exp_config["parallel_environments"])])
    training_env = VecNormalize(training_env, norm_obs=True, norm_reward=True,
                                gamma=experiment.exp_config["rl_algorithm"]["gamma"], training=True)

    experiment.model_type = algorithm_class
    with open(f"{experiment.experiment_folder_path}/experiment_object.pickle", "wb") as handle:
        pickle.dump(experiment, handle, protocol=pickle.HIGHEST_PROTOCOL)

    model = algorithm_class(policy=experiment.exp_config["rl_algorithm"]["policy"],  # MlpPolicy by default
                            env=training_env,
                            verbose=2,
                            seed=experiment.exp_config["random_seed"],
                            gamma=experiment.exp_config["rl_algorithm"]["gamma"],
                            tensorboard_log=args.logdir.format(args.exp_id),  # "tensor_log",
                            policy_kwargs=copy.copy(
                                experiment.exp_config["rl_algorithm"]["model_architecture"]
                            ),  # This is necessary because SB modifies the passed dict.
                            **experiment.exp_config["rl_algorithm"]["args"])
    logging.warning(
        f"Creating model with NN architecture(value/policy): {experiment.exp_config['rl_algorithm']['model_architecture']}")

    experiment.set_model(model)

    # get the performance of heuristic algorithms.
    logging.disable(logging.INFO)
    experiment.compare()
    logging.disable(logging.DEBUG)

    callback_test_env = VecNormalize(
        DummyVecEnv([experiment.make_env(0,
                                         environment_type=EnvironmentType.TESTING,
                                         workloads_in=None,
                                         db_config=experiment.schema.db_config)]),
        norm_obs=True,
        norm_reward=False,
        gamma=experiment.exp_config["rl_algorithm"]["gamma"],
        training=False)
    test_callback = EvalCallbackWithTBRunningAverage(
        n_eval_episodes=experiment.exp_config["workload"]["validation_testing"]["number_of_workloads"],
        eval_freq=round(experiment.exp_config["validation_frequency"] / experiment.exp_config["parallel_environments"]),
        eval_env=callback_test_env,
        verbose=1,
        name="test",
        deterministic=True,
        comparison_performances=experiment.comparison_performances["test"])

    callback_validation_env = VecNormalize(
        DummyVecEnv([experiment.make_env(0, environment_type=EnvironmentType.VALIDATION,
                                         workloads_in=None,
                                         db_config=experiment.schema.db_config)]),
        norm_obs=True,
        norm_reward=False,
        gamma=experiment.exp_config["rl_algorithm"]["gamma"],
        training=False)
    validation_callback = EvalCallbackWithTBRunningAverage(
        n_eval_episodes=experiment.exp_config["workload"]["validation_testing"]["number_of_workloads"],
        eval_freq=round(experiment.exp_config["validation_frequency"] / experiment.exp_config["parallel_environments"]),
        eval_env=callback_validation_env,
        best_model_save_path=experiment.experiment_folder_path,
        verbose=1,
        name="validation",
        deterministic=True,
        comparison_performances=experiment.comparison_performances["validation"])
    callbacks = [validation_callback, test_callback]

    if len(experiment.multi_validation_wl) > 0:
        callback_multi_validation_env = VecNormalize(
            DummyVecEnv([experiment.make_env(0, EnvironmentType.VALIDATION,
                                             experiment.multi_validation_wl,
                                             db_config=experiment.schema.db_config)]),
            norm_obs=True,
            norm_reward=False,
            gamma=experiment.exp_config["rl_algorithm"]["gamma"],
            training=False,
        )
        multi_validation_callback = EvalCallbackWithTBRunningAverage(
            n_eval_episodes=len(experiment.multi_validation_wl),
            eval_freq=round(
                experiment.exp_config["validation_frequency"] / experiment.exp_config["parallel_environments"]),
            eval_env=callback_multi_validation_env,
            best_model_save_path=experiment.experiment_folder_path,
            verbose=1,
            name="multi_validation",
            deterministic=True,
            comparison_performances={},
        )
        callbacks.append(multi_validation_callback)

    # set the `training_start_time`.
    experiment.record_learning_start_time()
    model.learn(total_timesteps=args.timesteps,
                callback=callbacks,
                tb_log_name=experiment.id)  # the name of the run for tensorboard log
    experiment.finish_learning_save_model(model.get_env(),  # training_env,
                                          validation_callback.moving_average_step * experiment.exp_config[
                                              "parallel_environments"],
                                          validation_callback.best_model_step * experiment.exp_config[
                                              "parallel_environments"])
    experiment.finish_evaluation()
