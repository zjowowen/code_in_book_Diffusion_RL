import random
import matplotlib
import numpy as np
from easydict import EasyDict
from rich.progress import track

# matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from easydict import EasyDict

from grl.generative_models.diffusion_model.diffusion_model import DiffusionModel
from grl.utils import set_seed
from grl.utils.log import log


def generate_samples(radius, num_samples, x_bias=0.5):
    left_sample = []
    right_sample = []
    and_sample = []
    for i in range(num_samples):
        accept_flag = False
        while not accept_flag:
            # 生成矩形内的随机点
            x = random.uniform(-radius - x_bias, radius + x_bias)
            y = random.uniform(-radius, radius)
            # 计算点到两个圆心的距离
            distance_2_left = (x + x_bias) ** 2 + y**2
            distance_2_right = (x - x_bias) ** 2 + y**2
            # 交集
            if distance_2_left <= radius**2 and distance_2_right <= radius**2:
                and_sample.append([x, y])
                accept_flag = True
            # 左圆
            if distance_2_left <= radius**2 and distance_2_right > radius**2:
                left_sample.append([x, y])
                accept_flag = True
            # 右圆
            if distance_2_left > radius**2 and distance_2_right <= radius**2:
                right_sample.append([x, y])
                accept_flag = True
    return np.array(left_sample), np.array(right_sample), np.array(and_sample)


def visualize_samples(samples_np_array, color, marker, label):
    x_coords = samples_np_array[:, 0]
    y_coords = samples_np_array[:, 1]
    plt.scatter(x_coords, y_coords, color=color, marker=marker, label=label, alpha=0.2)
    plt.title("Train DataSet")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")


def visualize_samples_seperate(samples_list, colors, markers, labels):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    for i, (samples, color, marker, label) in enumerate(
        zip(samples_list, colors, markers, labels)
    ):
        row = i // 2
        col = i % 2
        ax = axs[row, col]
        x_coords = samples[:, 0]
        y_coords = samples[:, 1]
        ax.scatter(
            x_coords, y_coords, color=color, marker=marker, label=label, alpha=0.2
        )
        ax.set_title(f"Plot {label} Result")
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.set_xlim((-3, 3))
        ax.set_ylim((-2, 2))
        ax.legend()
    plt.tight_layout()
    plt.show()


samples_num = 2000000
left_sample, right_sample, and_sample = generate_samples(1, samples_num, 0.5)
plt.figure(figsize=(15, 10))
plt.xlim((-3, 3))
plt.ylim((-2, 2))
visualize_samples(left_sample, "red", "o", "Left sample")
visualize_samples(right_sample, "blue", "s", "Right sample")
visualize_samples(and_sample, "green", "^", "And_sample")
plt.legend()
plt.show()


condition = np.zeros((samples_num, 2))
condition[: left_sample.shape[0], 0] = 1
condition[left_sample.shape[0] : left_sample.shape[0] + right_sample.shape[0], 1] = 1
condition[left_sample.shape[0] + right_sample.shape[0] :] = 1

train_data = np.vstack((left_sample, right_sample, and_sample))
train_data = np.hstack((train_data, condition))

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
x_size = 2
t_embedding_dim = 32
t_encoder = dict(
    type="GaussianFourierProjectionTimeEncoder",
    args=dict(
        embed_dim=t_embedding_dim,
        scale=30.0,
    ),
)
c_encoder = dict(
    type="GaussianFourierProjectionEncoder",
    args=dict(
        embed_dim=t_embedding_dim,
        scale=30.0,
        x_shape=[2],
    ),
)
config = EasyDict(
    dict(
        device=device,
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
                type="linear_vp_sde",
                beta_0=0.1,
                beta_1=20.0,
            ),
            model=dict(
                type="noise_function",
                args=dict(
                    t_encoder=t_encoder,
                    condition_encoder=c_encoder,
                    backbone=dict(
                        type="TemporalSpatialResidualNet",
                        args=dict(
                            hidden_sizes=[512, 256, 128],
                            t_dim=t_embedding_dim,
                            condition_dim=64,
                            condition_hidden_dim=32,
                            t_condition_hidden_dim=256,
                            output_dim=x_size,
                        ),
                    ),
                ),
            ),
        ),
        parameter=dict(
            lr=5e-3,
            data_num=1000000,
            iterations=2000,
            batch_size=4096,
            clip_grad_norm=1.0,
            eval_freq=500,
            checkpoint_freq=100,
            checkpoint_path="./checkpoint",
            video_save_path="./video",
            device=device,
        ),
    )
)


def get_train_data(dataloader):
    while True:
        yield from dataloader


seed_value = set_seed()
log.info(f"start exp with seed value {seed_value}.")
diffusion_model = DiffusionModel(config=config.diffusion_model).to(
    config.diffusion_model.device
)
diffusion_model = torch.compile(diffusion_model)
optimizer = torch.optim.Adam(
    diffusion_model.parameters(),
    lr=config.parameter.lr,
)

data_loader = torch.utils.data.DataLoader(
    train_data, batch_size=config.parameter.batch_size, shuffle=True
)
data_generator = get_train_data(data_loader)

p_uncon = 0.5
gradient_sum = 0.0
loss_sum = 0.0
counter = 0
iteration = 0
for iteration in track(range(config.parameter.iterations), description="Training"):
    batch_data = next(data_generator)
    batch_data = batch_data.to(config.device).float()
    diffusion_model.train()
    loss = (1 - p_uncon) * diffusion_model.score_matching_loss(
        batch_data[:, :2], batch_data[:, 2:]
    ) + p_uncon * diffusion_model.score_matching_loss(
        batch_data[:, :2], torch.zeros_like(batch_data[:, 2:]).to(batch_data)
    )
    optimizer.zero_grad()
    loss.backward()
    gradien_norm = torch.nn.utils.clip_grad_norm_(
        diffusion_model.parameters(), config.parameter.clip_grad_norm
    )
    optimizer.step()
    gradient_sum += gradien_norm.item()
    loss_sum += loss.item()
    counter += 1

num = 4000
con_00 = torch.zeros((num, 2))
con_01 = torch.zeros((num, 2))
con_10 = torch.zeros((num, 2))
con_11 = torch.zeros((num, 2))
con_01[:, 1] = 1
con_10[:, 0] = 1
con_11 += 1
diffusion_model.eval()
t_span = torch.linspace(0.0, 1.0, 1000)
x_t = (
    diffusion_model.sample_forward_process(
        t_span=t_span, condition=con_00.to(config.device).float()
    )
    .cpu()
    .detach()
)
res_00 = x_t.cpu().numpy()[-1]
t_span = torch.linspace(0.0, 1.0, 1000)
x_t = (
    diffusion_model.sample_forward_process(
        t_span=t_span, condition=con_01.to(config.device).float()
    )
    .cpu()
    .detach()
)
res_01 = x_t.cpu().numpy()[-1]
t_span = torch.linspace(0.0, 1.0, 1000)
x_t = (
    diffusion_model.sample_forward_process(
        t_span=t_span, condition=con_10.to(config.device).float()
    )
    .cpu()
    .detach()
)
res_10 = x_t.cpu().numpy()[-1]
t_span = torch.linspace(0.0, 1.0, 1000)
x_t = (
    diffusion_model.sample_forward_process(
        t_span=t_span, condition=con_11.to(config.device).float()
    )
    .cpu()
    .detach()
)
res_11 = x_t.cpu().numpy()[-1]
visualize_samples_seperate(
    [res_00, res_01, res_10, res_11],
    ["purple", "b", "r", "g"],
    ["*", "s", "o", "^"],
    ["OR", "RIGHT", "LEFT", "AND"],
)
