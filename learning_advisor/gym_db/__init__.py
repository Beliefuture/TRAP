from gym.envs.registration import register

register(id="DB-v1", entry_point="gym_db.envs:DBEnvV1")

register(id="DB-v3", entry_point="gym_db.envs:DBEnvV3")

register(id="DB-v4", entry_point="gym_db.envs:DBEnvV4")
