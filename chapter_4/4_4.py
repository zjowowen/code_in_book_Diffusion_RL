import os
import signal
import sys
from typing import Callable, List, Optional, Tuple, Union
import matplotlib
import numpy as np
from easydict import EasyDict
from rich.progress import track

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from diffusers.models import AutoencoderKL
from easydict import EasyDict
from matplotlib import animation
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from accelerate import Accelerator

from grl.generative_models.diffusion_model.diffusion_model import DiffusionModel
from grl.neural_network.encoders import ExponentialFourierProjectionTimeEncoder
from grl.neural_network.transformers.dit import DiTBlock
from grl.neural_network import register_module
from grl.utils.log import log


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Overview:
        Modulate the input tensor x with the shift and scale tensors.
    Arguments:
        x (:obj:`torch.Tensor`): The input tensor.
        shift (:obj:`torch.Tensor`): The shift tensor.
        scale (:obj:`torch.Tensor`): The scale tensor.
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_sincos_pos_embed_from_grid(
    embed_dim: int,
    pos: np.ndarray,
):
    """
    Overview:
        Get positional embeddings for 1D data.
    Arguments:
        embed_dim (:obj:`int`): The output dimension for each position.
        pos (:obj:`np.ndarray`): The input positions.
    Returns:
        emb (:obj:`np.ndarray`): The positional embeddings.
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    out = np.einsum("...,d->...d", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (M, D)
    return emb


def get_2d_pos_embed(
    embed_dim: int,
    grid_num: List[int],
):
    """
    Overview:
        Get 3D positional embeddings for 3D data.
    Arguments:
        embed_dim (:obj:`int`): The output dimension of embeddings for each grid.
        grid_num (:obj:`List[int]`): The number of the grid in each dimension.
    """
    assert len(grid_num) == 2
    grid_num_sum = grid_num[0] + grid_num[1]
    assert (
        embed_dim % grid_num_sum == 0
    ), f"Embedding dimension {embed_dim} must be divisible by the total grid size {grid_num_sum}."
    embed_dim_per_grid = embed_dim // grid_num_sum
    grid_0 = np.arange(grid_num[0], dtype=np.float32)
    grid_1 = np.arange(grid_num[1], dtype=np.float32)

    grid = np.meshgrid(grid_1, grid_0)  # here w goes first
    grid = np.stack([grid[1], grid[0]], axis=0)
    emb_0 = get_sincos_pos_embed_from_grid(embed_dim_per_grid * grid_num[0], grid[0])
    emb_1 = get_sincos_pos_embed_from_grid(embed_dim_per_grid * grid_num[1], grid[1])

    # emb is of shape (total_grid_num, embed_dim)
    emb = np.concatenate([emb_0, emb_1], axis=-1)
    return emb


class FinalLayer2D(nn.Module):
    """
    Overview:
        The final layer of DiT for 2D data.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
        self,
        hidden_size: int,
        patch_size: Union[int, List[int], Tuple[int]],
        out_channels: Union[int, List[int], Tuple[int]],
    ):
        """
        Overview:
            Initialize the final layer.
        Arguments:
            hidden_size (:obj:`int`): The hidden size.
            patch_size (:obj:`Union[int, List[int], Tuple[int]]`): The patch size of each token in attention layer.
            out_channels (:obj:`Union[int, List[int], Tuple[int]]`): The number of output channels.
        """
        super().__init__()
        assert (
            isinstance(patch_size, (list, tuple))
            and len(patch_size) == 2
            or isinstance(patch_size, int)
        )
        if isinstance(patch_size, int):
            self.patch_size = [patch_size] * 2
        else:
            self.patch_size = list(patch_size)
        assert isinstance(out_channels, (list, tuple)) or isinstance(out_channels, int)
        if isinstance(out_channels, int):
            self.out_channels = [out_channels]
        else:
            self.out_channels = list(out_channels)

        output_dim = np.prod(self.patch_size) * np.prod(self.out_channels)

        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """
        Overview:
            Forward pass of the final layer.
        Arguments:
            x (:obj:`torch.Tensor`): The input tensor of shape (N, total_patches, hidden_size).
            c (:obj:`torch.Tensor`): The conditioning tensor.
        Returns:
            x (:obj:`torch.Tensor`): The output tensor of shape (N, total_patches, patch_size[0] * patch_size[1] * **out_channels).
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class Patchify2D(nn.Module):
    """
    Overview:
        Patchify the input tensor of shape (H, W) of attention layer.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
        self,
        channel_size: Union[int, List[int]] = [3],
        data_size: List[int] = [32, 32],
        patch_size: List[int] = [2, 2],
        hidden_size: int = 768,
        bias: bool = False,
        convolved: bool = False,
    ):
        """
        Overview:
            Initialize the patchify layer.
        Arguments:
            channel_size (:obj:`Union[int, List[int]]`): The number of input channels, defaults to 3.
            data_size (:obj:`List[int]`): The input size of data, defaults to [32, 32, 32].
            patch_size (:obj:`List[int]`): The patch size of each token for attention layer, defaults to [2, 2, 2].
            hidden_size (:obj:`int`): The hidden size of attention layer, defaults to 768.
            bias (:obj:`bool`): Whether to use bias, defaults to False.
            convolved (:obj:`bool`): Whether to use fully connected layer for all channels, defaults to False.
        """
        super().__init__()
        assert isinstance(data_size, (list, tuple)) or isinstance(data_size, int)
        self.channel_size = (
            list(channel_size)
            if isinstance(channel_size, (list, tuple))
            else [channel_size]
        )
        self.patch_size = patch_size

        in_channels = 1
        for i in self.channel_size:
            in_channels *= i

        self.num_patches = 1
        for i in range(2):
            self.num_patches *= data_size[i] // patch_size[i]

        if convolved:
            self.proj = nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_size,
                kernel_size=patch_size,
                stride=patch_size,
                bias=bias,
            )
        else:
            self.proj = nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_size,
                kernel_size=patch_size,
                stride=patch_size,
                groups=in_channels,
                bias=bias,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Forward pass of the patchify layer.
        Arguments:
            x (:obj:`torch.Tensor`): The input tensor of shape (B, C, H, W).
        Returns:
            x (:obj:`torch.Tensor`): The output tensor of shape (B, H'* W', hidden_size). \
            where H' = H // patch_size[0], W' = W // patch_size[1].
        """

        # x: (B, (C1, C2), H, W) # x.reshape(shape=(x.shape[0], *self.channel_size, x.shape[-2], x.shape[-1]))
        x = x.flatten(start_dim=1, end_dim=-3)
        # x: (B, C1 * C2, H, W)
        x = self.proj(x)
        return x


class DiT2D(nn.Module):
    """
    Overview:
        Transformer backbone for Diffusion model for data of 2D shape.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
        self,
        patch_block_size: Union[List[int], Tuple[int]] = [32, 32],
        patch_size: Union[int, List[int], Tuple[int]] = 2,
        in_channels: Union[int, List[int], Tuple[int]] = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        convolved: bool = False,
    ):
        """
        Overview:
            Initialize the DiT model.
        Arguments:
            patch_block_size (:obj:`Union[List[int], Tuple[int]]`): The size of patch block, defaults to [10, 32, 32].
            patch_size (:obj:`Union[int, List[int], Tuple[int]]`): The patch size of each token in attention layer, defaults to 2.
            in_channels (:obj:`Union[int, List[int], Tuple[int]]`): The number of input channels, defaults to 4.
            hidden_size (:obj:`int`): The hidden size of attention layer, defaults to 1152.
            depth (:obj:`int`): The depth of transformer, defaults to 28.
            num_heads (:obj:`int`): The number of attention heads, defaults to 16.
            mlp_ratio (:obj:`float`): The hidden size of the MLP with respect to the hidden size of Attention, defaults to 4.0.
            convolved (:obj:`bool`): Whether to use fully connected layer for all channels, defaults to False.
        """
        super().__init__()

        assert (
            isinstance(patch_block_size, (list, tuple))
            and len(patch_block_size) == 2
            or isinstance(patch_block_size, int)
        )
        self.patch_block_size = (
            list(patch_block_size)
            if isinstance(patch_block_size, (list, tuple))
            else [patch_block_size] * 2
        )
        assert (
            isinstance(patch_size, (list, tuple))
            and len(patch_size) == 2
            or isinstance(patch_size, int)
        )
        self.patch_size = (
            list(patch_size)
            if isinstance(patch_size, (list, tuple))
            else [patch_size] * 2
        )
        for i in range(2):
            assert (
                self.patch_block_size[i] % self.patch_size[i] == 0
            ), f"Patch block size {self.patch_block_size[i]} should be divisible by patch size {self.patch_size[i]}."
        self.patch_grid_num = [
            self.patch_block_size[i] // self.patch_size[i] for i in range(2)
        ]

        assert isinstance(in_channels, (list, tuple)) or isinstance(in_channels, int)
        self.in_channels = (
            list(in_channels)
            if isinstance(in_channels, (list, tuple))
            else [in_channels]
        )
        self.out_channels = self.in_channels

        self.num_heads = num_heads

        self.x_embedder = Patchify2D(
            in_channels,
            patch_block_size,
            patch_size,
            hidden_size,
            bias=True,
            convolved=convolved,
        )
        self.t_embedder = ExponentialFourierProjectionTimeEncoder(hidden_size)

        pos_embed = get_2d_pos_embed(
            embed_dim=hidden_size, grid_num=self.patch_grid_num
        )
        self.pos_embed = nn.Parameter(
            torch.from_numpy(pos_embed).float(), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer2D(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        """
        Overview:
            Initialize the weights of the model.
        """

        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        Overview:
            Unpatchify the output tensor of attention layer.
        Arguments:
            x (:obj:`torch.Tensor`): The input tensor of shape (N, total_patches = H' * W', patch_size[0] * patch_size[1] * C)
        Returns:
            x (:obj:`torch.Tensor`): The output tensor of shape (N, T, C, H, W).
        """

        x = x.reshape(
            shape=(
                x.shape[0],
                self.patch_grid_num[0],
                self.patch_grid_num[1],
                self.patch_size[0],
                self.patch_size[1],
                np.prod(self.out_channels),
            )
        )
        x = torch.einsum("nhwqr...->n...hqwr", x)
        x = x.reshape(
            shape=(
                x.shape[0],
                *self.out_channels,
                self.patch_grid_num[0] * self.patch_size[0],
                self.patch_grid_num[1] * self.patch_size[1],
            )
        )

        return x

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ):
        """
        Overview:
            Forward pass of DiT for 3D data.
        Arguments:
            t (:obj:`torch.Tensor`): Tensor of diffusion timesteps.
            x (:obj:`torch.Tensor`): Tensor of inputs with spatial information (originally at t=0 it is tensor of videos or latent representations of videos).
            condition (:obj:`torch.Tensor`, optional): The input condition, such as class labels.
        """

        # x is of shape (N, C, H, W)
        x = self.x_embedder(x) + torch.einsum("HWh->hHW", self.pos_embed)
        x = x.reshape(shape=(x.shape[0], x.shape[1], -1))
        x = torch.einsum(
            "nhs->nsh", x
        )  # (N, total_patches, hidden_size), where total_patches = H' * W' = H * W / patch_size[0] * patch_size[1]
        t = self.t_embedder(t)  # (N, hidden_size)

        if condition is not None:
            # TODO: polish this part
            y = self.y_embedder(condition, self.training)  # (N, hidden_size)
            c = t + y  # (N, hidden_size)
        else:
            c = t

        for block in self.blocks:
            x = block(x, c)  # (N, total_patches, hidden_size)
        x = self.final_layer(
            x, c
        )  # (N, total_patches, patch_size[0] * patch_size[1] * C)
        x = self.unpatchify(x)  # (N, C, H, W)
        return x


register_module(DiT2D, "DiT2D_Special")


def make_config(device):

    image_size_H = 640
    image_size_W = 360
    channel_size = 3
    x_size = (channel_size, image_size_W, image_size_H)
    patch_block_size = (image_size_W, image_size_H)
    patch_size = (8, 8)
    patch_grid_num = (
        image_size_W // patch_size[0],
        image_size_H // patch_size[1],
    )
    num_heads = 6
    hidden_size = np.sum(patch_grid_num) * num_heads * 2
    assert (
        hidden_size % (np.sum(patch_grid_num) * num_heads) == 0
    ), f"hidden_size must be divisible by patch_grid_num * 2 * num_heads."

    config = EasyDict(
        dict(
            device=device,
            project="dit-minecraft-without-vae",
            data=dict(
                data_path="/root/minecraft/minecraft_data",
            ),
            model=dict(
                diffusion_model=dict(
                    device=device,
                    x_size=x_size,
                    alpha=1.0,
                    solver=dict(
                        type="ODESolver",
                        args=dict(
                            library="torchdiffeq",
                        ),
                    ),
                    path=dict(
                        type="linear_vp_sde",
                        beta_0=0.1,
                        beta_1=20.0,
                    ),
                    model=dict(
                        type="velocity_function",
                        args=dict(
                            backbone=dict(
                                type="DiT2D_Special",
                                args=dict(
                                    patch_block_size=patch_block_size,
                                    patch_size=patch_size,
                                    in_channels=channel_size,
                                    hidden_size=hidden_size,
                                    depth=6,
                                    num_heads=num_heads,
                                    convolved=True,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            parameter=dict(
                training_loss_type="flow_matching",
                batch_size=8,
                eval_freq=1,
                learning_rate=5e-4,
                iterations=200000,
                clip_grad_norm=1.0,
                checkpoint_freq=1,
                checkpoint_path="./checkpoint-mc",
                image_save_path="./images-mc",
            ),
        )
    )

    return config


if __name__ == "__main__":

    accelerator = Accelerator()
    device = accelerator.device
    config = make_config(device)

    diffusion_model = DiffusionModel(config=config.model.diffusion_model).to(device)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            ),
        ]
    )

    dataset = ImageFolder(config.data.data_path, transform=transform)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.parameter.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    optimizer = torch.optim.Adam(
        diffusion_model.parameters(),
        lr=config.parameter.learning_rate,
    )

    if os.path.exists(config.parameter.checkpoint_path):
        checkpoint_files = sorted(
            [
                f
                for f in os.listdir(config.parameter.checkpoint_path)
                if f.endswith(".pt")
            ],
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
        )
        if not checkpoint_files:
            pass
        else:
            checkpoint_file = os.path.join(
                config.parameter.checkpoint_path, checkpoint_files[-1]
            )
            with accelerator.main_process_first():
                checkpoint = torch.load(
                    config.parameter.checkpoint_path, map_location="cpu"
                )
                ori_state_dict = {
                    k.replace("module.", ""): v for k, v in checkpoint["model"].items()
                }
                ori_state_dict = {
                    k.replace("_orig_mod.", ""): v for k, v in ori_state_dict.items()
                }
                diffusion_model.model.load_state_dict(ori_state_dict)

        diffusion_model.model, optimizer, data_loader = accelerator.prepare(
            diffusion_model.model, optimizer, data_loader
        )

    gradient_sum = 0
    loss_sum = 0
    counter = 0

    for iteration in range(config.parameter.iterations):

        if (
            iteration > 0 and iteration % config.parameter.eval_freq == 0
        ) or iteration == config.parameter.iterations - 1:
            diffusion_model.eval()
            t_span = torch.linspace(0.0, 1.0, 1000)
            x_t = diffusion_model.sample_forward_process(
                t_span=t_span, batch_size=1
            ).detach()
            x_t = torch.einsum("tbchw->btchw", x_t)
            x_t = accelerator.gather_for_metrics(x_t)
            x_t = torch.einsum("btchw->tbchw", x_t)
            x_t = [
                x.squeeze(0) for x in torch.split(x_t, split_size_or_sections=1, dim=0)
            ]
            image_x_1 = x_t[-1]
            if accelerator.is_local_main_process:
                if not os.path.exists(config.parameter.image_save_path):
                    os.makedirs(config.parameter.image_save_path)

                torchvision.utils.save_image(
                    image_x_1,
                    os.path.join(
                        config.parameter.image_save_path, f"iteration_{iteration}.png"
                    ),
                    nrow=4,
                    normalize=True,
                    value_range=(-1, 1),
                )

        for batch_data, label in track(
            data_loader,
            description="Training",
            disable=not accelerator.is_local_main_process,
        ):

            batch_data = batch_data.to(config.device)

            diffusion_model.train()
            if config.parameter.training_loss_type == "flow_matching":
                loss = diffusion_model.flow_matching_loss(batch_data)
            elif config.parameter.training_loss_type == "score_matching":
                loss = diffusion_model.score_matching_loss(batch_data)
            else:
                raise NotImplementedError("Unknown loss type")
            optimizer.zero_grad()
            accelerator.backward(loss)
            gradien_norm = torch.nn.utils.clip_grad_norm_(
                diffusion_model.parameters(), config.parameter.clip_grad_norm
            )
            optimizer.step()
            gradient_sum += gradien_norm.item()
            loss_sum += loss.item()
            counter += 1

            log.info(
                f"iteration {iteration}, gradient {gradient_sum/counter}, loss {loss_sum/counter}"
            )

        if (iteration + 1) % config.parameter.checkpoint_freq == 0:
            if accelerator.is_main_process:
                if not os.path.exists(config.parameter.checkpoint_path):
                    os.makedirs(config.parameter.checkpoint_path)
            accelerator.wait_for_everyone()
            accelerator.save_state(
                os.path.join(
                    config.parameter.checkpoint_path, f"checkpoint_{iteration}"
                )
            )
