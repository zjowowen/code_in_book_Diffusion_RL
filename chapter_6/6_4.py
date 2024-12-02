import gymnasium as gym
import os
import cv2
from typing import List, Optional
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from easydict import EasyDict
from rich.progress import track

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

import torch._dynamo

torch._dynamo.config.suppress_errors = True

from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
import treetensor
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from accelerate import Accelerator

from matplotlib import animation

from grl.generative_models.diffusion_model.diffusion_model import DiffusionModel
from grl.utils.log import log
from grl.neural_network import register_module
from grl.datasets.gp import GPAtariVisualTensorDictDataset

from functools import partial


def ramdon_action(env):
    action = env.action_space.sample()
    return action


def collect_data(env_id, origin_image_size):

    obs_list = []
    reward_list = []
    action_list = []
    next_obs_list = []
    done_list = []

    env = gym.make(env_id, render_mode="rgb_array")
    obs = env.reset()

    counter = 0

    terminated, truncated = False, False

    while not terminated and not truncated:
        action = ramdon_action(env)

        if isinstance(obs, tuple):
            obs = obs[0]
        else:
            obs = obs
        assert obs.shape == (
            origin_image_size[0],
            origin_image_size[1],
            origin_image_size[2],
        ), "obs shape is not correct"
        obs_list.append(torch.tensor(obs, dtype=torch.uint8).unsqueeze(0))

        obs, reward, terminated, truncated, info = env.step(action)

        reward_list.append(torch.tensor(reward, dtype=torch.float32))
        action_list.append(torch.tensor(action, dtype=torch.uint8))
        assert obs.shape == (
            origin_image_size[0],
            origin_image_size[1],
            origin_image_size[2],
        ), "obs shape is not correct"
        next_obs_list.append(torch.tensor(obs, dtype=torch.uint8).unsqueeze(0))
        done_list.append(torch.tensor(terminated or truncated, dtype=torch.uint8))

        counter += 1

    env.close()

    obs_tensor = torch.concatenate(obs_list)
    reward_tensor = torch.tensor(reward_list)
    action_tensor = torch.tensor(action_list)
    next_obs_tensor = torch.concatenate(next_obs_list)
    done_tensor = torch.tensor(done_list)

    return obs_tensor, reward_tensor, action_tensor, next_obs_tensor, done_tensor


def resize_image_torch(image, size):
    # image is numpy array of shape (210, 160, 3), resize to (image_size, image_size, 3)
    image = image.to(torch.float32)
    image = image.permute(0, 3, 1, 2)  # Now shape is (B, 3, 210, 160)
    # Define the target size
    target_size = (size, size)
    # Use torch.nn.functional.interpolate to resize with bicubic interpolation
    image = F.interpolate(image, size=target_size, mode="bicubic", align_corners=False)
    image = image[:, [2, 1, 0], :, :].contiguous().clamp(0, 255)
    image = image.to(torch.uint8).to(torch.float32)
    image = transform_obs(image)
    return image


def transform_obs(obs):
    obs = obs - 127.5  # [0, 255] -> [-127.5, 127.5]
    obs = obs / 127.5  # [-127.5, 127.5] -> [-1, 1]
    return obs


GN_GROUP_SIZE = 32
GN_EPS = 1e-5
ATTN_HEAD_DIM = 8

Conv1x1 = partial(nn.Conv2d, kernel_size=1, stride=1, padding=0)
Conv3x3 = partial(nn.Conv2d, kernel_size=3, stride=1, padding=1)


class GroupNorm(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        num_groups = max(1, in_channels // GN_GROUP_SIZE)
        self.norm = nn.GroupNorm(num_groups, in_channels, eps=GN_EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class AdaGroupNorm(nn.Module):
    def __init__(self, in_channels: int, cond_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_groups = max(1, in_channels // GN_GROUP_SIZE)
        self.linear = nn.Linear(cond_channels, in_channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        assert x.size(1) == self.in_channels
        x = F.group_norm(x, self.num_groups, eps=GN_EPS)
        scale, shift = self.linear(cond)[:, :, None, None].chunk(2, dim=1)
        return x * (1 + scale) + shift


class SelfAttention2d(nn.Module):
    def __init__(self, in_channels: int, head_dim: int = ATTN_HEAD_DIM) -> None:
        super().__init__()
        self.n_head = max(1, in_channels // head_dim)
        assert in_channels % self.n_head == 0
        self.norm = GroupNorm(in_channels)
        self.qkv_proj = Conv1x1(in_channels, in_channels * 3)
        self.out_proj = Conv1x1(in_channels, in_channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.qkv_proj(x)
        qkv = (
            qkv.view(n, self.n_head * 3, c // self.n_head, h * w)
            .transpose(2, 3)
            .contiguous()
        )
        q, k, v = [x for x in qkv.chunk(3, dim=1)]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(2, 3).reshape(n, c, h, w)
        return x + self.out_proj(y)


class ResBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, cond_channels: int, attn: bool
    ) -> None:
        super().__init__()
        should_proj = in_channels != out_channels
        self.proj = Conv1x1(in_channels, out_channels) if should_proj else nn.Identity()
        self.norm1 = AdaGroupNorm(in_channels, cond_channels)
        self.conv1 = Conv3x3(in_channels, out_channels)
        self.norm2 = AdaGroupNorm(out_channels, cond_channels)
        self.conv2 = Conv3x3(out_channels, out_channels)
        self.attn = SelfAttention2d(out_channels) if attn else nn.Identity()
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        r = self.proj(x)
        x = self.conv1(F.silu(self.norm1(x, cond)))
        x = self.conv2(F.silu(self.norm2(x, cond)))
        x = x + r
        x = self.attn(x)
        return x


class ResBlocks(nn.Module):
    def __init__(
        self,
        list_in_channels: List[int],
        list_out_channels: List[int],
        cond_channels: int,
        attn: bool,
    ) -> None:
        super().__init__()
        assert len(list_in_channels) == len(list_out_channels)
        self.in_channels = list_in_channels[0]
        self.resblocks = nn.ModuleList(
            [
                ResBlock(in_ch, out_ch, cond_channels, attn)
                for (in_ch, out_ch) in zip(list_in_channels, list_out_channels)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        to_cat: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        outputs = []
        for i, resblock in enumerate(self.resblocks):
            x = x if to_cat is None else torch.cat((x, to_cat[i]), dim=1)
            x = resblock(x, cond)
            outputs.append(x)
        return x, outputs


class Downsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=2, padding=1
        )
        nn.init.orthogonal_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = Conv3x3(in_channels, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        cond_channels: int,
        depths: List[int],
        channels: List[int],
        attn_depths: List[int],
    ) -> None:
        super().__init__()
        assert len(depths) == len(channels) == len(attn_depths)

        d_blocks, u_blocks = [], []
        for i, n in enumerate(depths):
            c1 = channels[max(0, i - 1)]
            c2 = channels[i]
            d_blocks.append(
                ResBlocks(
                    list_in_channels=[c1] + [c2] * (n - 1),
                    list_out_channels=[c2] * n,
                    cond_channels=cond_channels,
                    attn=attn_depths[i],
                )
            )
            u_blocks.append(
                ResBlocks(
                    list_in_channels=[2 * c2] * n + [c1 + c2],
                    list_out_channels=[c2] * n + [c1],
                    cond_channels=cond_channels,
                    attn=attn_depths[i],
                )
            )
        self.d_blocks = nn.ModuleList(d_blocks)
        self.u_blocks = nn.ModuleList(reversed(u_blocks))

        self.mid_blocks = ResBlocks(
            list_in_channels=[channels[-1]] * 2,
            list_out_channels=[channels[-1]] * 2,
            cond_channels=cond_channels,
            attn=True,
        )

        downsamples = [nn.Identity()] + [Downsample(c) for c in channels[:-1]]
        upsamples = [nn.Identity()] + [Upsample(c) for c in reversed(channels[:-1])]
        self.downsamples = nn.ModuleList(downsamples)
        self.upsamples = nn.ModuleList(upsamples)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        d_outputs = []
        for block, down in zip(self.d_blocks, self.downsamples):
            x_down = down(x)
            x, block_outputs = block(x_down, cond)
            d_outputs.append((x_down, *block_outputs))

        x, _ = self.mid_blocks(x, cond)

        u_outputs = []
        for block, up, skip in zip(self.u_blocks, self.upsamples, reversed(d_outputs)):
            x_up = up(x)
            x, block_outputs = block(x_up, cond, skip[::-1])
            u_outputs.append((x_up, *block_outputs))

        return x, d_outputs, u_outputs


class MyModule(nn.Module):

    def __init__(self):
        super(MyModule, self).__init__()
        self.unet = UNet(
            cond_channels=256,
            depths=[2, 2, 2, 2],
            channels=[64, 64, 64, 64],
            attn_depths=[0, 0, 0, 0],
        )

        num_actions = 9
        cond_channels = 256
        num_steps_conditioning = 1

        self.conv_in = Conv3x3((num_steps_conditioning + 1) * 3, 64)
        self.norm_out = GroupNorm(64)

        self.conv_out = Conv3x3(64, 3)

        self.act_emb = nn.Sequential(
            nn.Embedding(num_actions, cond_channels // num_steps_conditioning // 2),
            nn.Flatten(),  # b t e -> b (t e)
        )

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition: torch.Tensor = None,
    ) -> torch.Tensor:
        if condition is not None:
            action_emb = self.act_emb(condition["action"].unsqueeze(1).to(torch.long))
            cond = torch.cat((action_emb, t), dim=1)

            # concatenate condition and t

            x = torch.cat((x, condition["state"]), dim=1)
            x = x.reshape(x.shape[0], -1, x.shape[2], x.shape[3])
        else:
            cond = t
        x = self.conv_in(x)
        x, d_outputs, u_outputs = self.unet(cond=cond, x=x)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return x


register_module(MyModule, "MyModule")


def make_config(device):

    image_size = 256
    x_size = (3, 256, 256)
    data_num = 100000
    t_embedding_dim = 128
    t_encoder = dict(
        type="GaussianFourierProjectionTimeEncoder",
        args=dict(
            embed_dim=t_embedding_dim,
            scale=30.0,
        ),
    )
    config = EasyDict(
        dict(
            device=device,
            dataset=dict(
                image_size=image_size,
                image_path="./images/",
                origin_image_size=(210, 160, 3),
            ),
            diffusion_model=dict(
                device=device,
                x_size=x_size,
                alpha=1.0,
                solver=dict(
                    type="ODESolver",
                    args=dict(
                        library="torchdyn",
                    ),
                ),
                path=dict(
                    type="gvp",
                ),
                model=dict(
                    type="velocity_function",
                    args=dict(
                        t_encoder=t_encoder,
                        backbone=dict(
                            type="MyModule",
                            args={},
                        ),
                    ),
                ),
            ),
            parameter=dict(
                training_loss_type="flow_matching",
                lr=5e-4,
                data_num=data_num,
                iterations=200000,
                batch_size=40,
                eval_freq=100,
                video_save_path="./video-atari-world-model",
                device=device,
            ),
        )
    )

    return config


def main():

    accelerator = Accelerator()
    device = accelerator.device
    config = make_config(device=device)

    log.info("config: \n{}".format(config))
    dataset = GPAtariVisualTensorDictDataset()
    env_id = "ale_py:ALE/MsPacman-v5"

    for i in range(10):
        obs_tensor, reward_tensor, action_tensor, next_obs_tensor, done_tensor = (
            collect_data(env_id, config.dataset.origin_image_size)
        )
        dataset.extend_data(
            [obs_tensor, action_tensor, done_tensor, next_obs_tensor, reward_tensor]
        )

    diffusion_model = DiffusionModel(config=config.diffusion_model).to(
        config.diffusion_model.device
    )

    optimizer = torch.optim.Adam(
        diffusion_model.model.parameters(),
        lr=config.parameter.lr,
    )

    diffusion_model.model, optimizer = accelerator.prepare(
        diffusion_model.model, optimizer
    )

    counter = 0
    iteration = 0

    def render_video(
        data_list, video_save_path, iteration, fps=100, dpi=100, prefix=""
    ):
        if not os.path.exists(video_save_path):
            os.makedirs(video_save_path)
        fig = plt.figure(figsize=(12, 12))

        ims = []

        for i, data in enumerate(data_list):

            grid = (
                make_grid(
                    data[:, [2, 1, 0], :, :].contiguous().clip(-1, 1),
                    value_range=(-1, 1),
                    padding=0,
                    nrow=4,
                )
                / 2
                + 0.5
            )
            img = ToPILImage()(grid)
            im = plt.imshow(img)
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True)
        ani.save(
            os.path.join(video_save_path, f"{prefix}_{iteration}.mp4"),
            fps=fps,
            dpi=dpi,
        )
        # clean up
        plt.close(fig)
        plt.clf()

    for iteration in range(config.parameter.iterations):

        if iteration >= 0 and iteration % config.parameter.eval_freq == 0:

            # random sample a batch of data
            random_idx = torch.randint(0, dataset.len, (4,))
            batch_data = dataset[random_idx]

            obs = batch_data["s"].to(config.device)
            obs = resize_image_torch(obs, config.dataset.image_size)
            action = batch_data["a"].to(config.device)
            next_obs = batch_data["s_"].to(config.device)
            next_obs = resize_image_torch(next_obs, config.dataset.image_size)

            diffusion_model.eval()

            t_span = torch.linspace(0.0, 1.0, 100)

            frame = []
            obs_temp = obs
            frame.append(accelerator.gather_for_metrics(obs_temp))

            for idx in range(20):

                condition = treetensor.torch.tensor(
                    dict(action=action, state=obs_temp)
                ).to(config.device)

                x_0 = torch.rand_like(obs)
                x_t = diffusion_model.sample_forward_process(
                    t_span=t_span, x_0=x_0, condition=condition
                )
                obs_temp = x_t[-1]
                frame.append(accelerator.gather_for_metrics(obs_temp).detach().cpu())

            frames = [x.squeeze(1) for x in frame]
            render_video(
                frames,
                config.parameter.video_save_path,
                iteration,
                fps=100,
                dpi=100,
                prefix="generate",
            )

        replay_buffer = TensorDictReplayBuffer(
            storage=dataset.storage,
            batch_size=config.parameter.batch_size,
            sampler=SamplerWithoutReplacement(),
            prefetch=10,
            pin_memory=True,
        )

        diffusion_model.train()

        for index, data in track(
            enumerate(replay_buffer),
            description=f"Epoch {iteration}",
            disable=not accelerator.is_local_main_process,
        ):
            obs = data["s"].to(config.device)
            obs = resize_image_torch(obs, config.dataset.image_size)
            action = data["a"].to(config.device)
            next_obs = data["s_"].to(config.device)
            next_obs = resize_image_torch(next_obs, config.dataset.image_size)
            condition = treetensor.torch.tensor(dict(action=action, state=obs)).to(
                config.device
            )

            if config.parameter.training_loss_type == "flow_matching":
                loss = diffusion_model.flow_matching_loss(
                    x=next_obs, condition=condition
                )
            elif config.parameter.training_loss_type == "score_matching":
                loss = diffusion_model.score_matching_loss(
                    x=next_obs, condition=condition
                )
            else:
                raise NotImplementedError("Unknown loss type")
            accelerator.backward(loss)
            counter += 1
            optimizer.step()
            optimizer.zero_grad()

            log.info(f"iteration {iteration}, step {counter}, loss {loss.item()}")


if __name__ == "__main__":
    main()
