import gym

from grl.algorithms.qgpo import QGPOAlgorithm
from grl.datasets import QGPOCustomizedTensorDictDataset
from grl.utils.log import log
from grl_pipelines.diffusion_model.configurations.lunarlander_continuous_qgpo import (
    config,
)


def qgpo_pipeline(config):
    qgpo = QGPOAlgorithm(
        config, dataset=QGPOCustomizedTensorDictDataset(numpy_data_path="./data.npz")
    )
    qgpo.train()

    agent = qgpo.deploy()
    env = gym.make(config.deploy.env.env_id)
    observation = env.reset()
    for _ in range(config.deploy.num_deploy_steps):
        env.render()
        observation, reward, done, _ = env.step(agent.act(observation))


if __name__ == "__main__":
    log.info("config: \n{}".format(config))
    qgpo_pipeline(config)
