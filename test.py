from read_example import plot_tf_trials
# --------------------------------------------------
# Define Training Config
# --------------------------------------------------
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # 图像尺寸
    image_size = (64, 64)
    # 训练批次大小
    train_batch_size = 36
    # 评估批次大小 
    eval_batch_size = 36
    # 训练轮数
    num_epochs = 60
    # 梯度累积步数（累计几次梯度更新一次参数）
    gradient_accumulation_steps = 1
    # 学习率
    learning_rate = 1e-4
    # 学习率衰减
    lr_warmup_steps = 500
    
    save_image_epochs = 10
    save_model_epochs = 20
    
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "output"
    # 是否上传模型到HF Hub
    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_model_id = "hibiscus/test_model"  # the name of the repository to create on the HF Hub
    hub_private_repo = None
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 42
    
config = TrainingConfig()


# --------------------------------------------------
# Resize to fit to UNet2DModel
# --------------------------------------------------
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

preprocess = transforms.Compose([
    transforms.Resize(config.image_size),
])

def transform_image(img):
    return preprocess(img)

test_data = np.load('data/tf_data.npz')
X_cwt = test_data['X_tf']
y = test_data['y']

X_cwt = torch.tensor(X_cwt, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
print("转换前的维度：", X_cwt.shape)

X_train_cwt_norm = transform_image(X_cwt)
print("转换后的维度：", X_train_cwt_norm.shape)


# --------------------------------------------------
# Create a UNet2DModel
# --------------------------------------------------
from diffusers import UNet2DModel

model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=5,  # the number of input channels, 3 for RGB images
    out_channels=5,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)
# print(model)


# --------------------------------------------------
# Test Model & visualization
# --------------------------------------------------
from PIL import Image
from diffusers import DDPMScheduler
import matplotlib.pyplot as plt

sample_channel = 0
sample_image = X_train_cwt_norm[0].unsqueeze(0)
print("输入sample图像维度：", sample_image.shape)
print("输出sample图像维度:", model(sample_image, timestep=0).sample.shape)

project_pic_dir = 'C:\\Users\\86136\\Desktop\\毕业设计\\picture'
test_noise_steps = 100

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
noise = torch.randn(sample_image.shape)
timesteps = torch.LongTensor([test_noise_steps])
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)


for i in range(0, 10):
    timesteps = torch.LongTensor([test_noise_steps*i])
    noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)
    plot_tf_trials(noisy_image, [0])



# # 存储每一个下标时的图像正态分布相似度，用于绘制三维图像，x轴为时间步，y轴为均值，z轴为方差
# junzhi_list = []
# fangcha_list = []
# timestep_list = []

# """for sample_pos in range(0, 1):
#     sample_image = X_train_cwt_norm[sample_pos].unsqueeze(0)
#     # 每加噪100步的图像，画出一张图，一共10张
#     fig, axs = plt.subplots(1, 10, figsize=(20, 2))
#     for i in range(0, 10):
#         timesteps = torch.LongTensor([test_noise_steps*i])
#         noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)
#         axs[i].imshow(noisy_image[0, 0, :, :].detach().numpy(), aspect='auto')
#         axs[i].axis("off")
#         timestep_list.append(i * 100)
#         junzhi_list.append(torch.mean(noisy_image).item())
#         fangcha_list.append(torch.var(noisy_image).item())
#         # 计算当前分布与正态噪声的相似度
#         axs[i].set_title(f"Step {i * 100} \n 均值：{torch.mean(noisy_image).item():.2f} \n 方差：{torch.var(noisy_image).item():.2f}")
#     plt.savefig("./tmpPic/sample_{}_noise_distribution.png".format(sample_pos), dpi=300)
#     plt.show()"""

# for sample_pos in range(0, X_train_cwt_norm.shape[0]):
#     sample_image = X_train_cwt_norm[sample_pos].unsqueeze(0)
#     for i in range(0, 10):
#         timesteps = torch.LongTensor([test_noise_steps*i])
#         noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)
#         timestep_list.append(i * 100)
#         junzhi_list.append(torch.mean(noisy_image).item())
#         fangcha_list.append(torch.var(noisy_image).item())
    
# # 画出三维图像
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(timestep_list, junzhi_list, fangcha_list)
# ax.set_xlabel('时间步')
# ax.set_ylabel('均值')
# ax.set_zlabel('方差')
# # plt.savefig("./tmpPic/sample_noise_distribution_3d.png", dpi=300, alpha=0.5)
# plt.show()




# --------------------------------------------------
# Train
# --------------------------------------------------
# import os
# from pathlib import Path
# from accelerate import Accelerator
# from torch.utils.data import DataLoader, Dataset
# import torch.nn.functional as F

# from diffusers.optimization import get_cosine_schedule_with_warmup
# from diffusers import DDPMPipeline
# from diffusers.utils import make_image_grid
# from diffusers import DDPMScheduler

# from huggingface_hub import create_repo, upload_folder
# from tqdm.auto import tqdm

# noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
# train_dataloader = DataLoader(X_train_cwt_norm, batch_size=config.train_batch_size, shuffle=True)
# optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
# lr_scheduler = get_cosine_schedule_with_warmup(
#     optimizer=optimizer,
#     num_warmup_steps=config.lr_warmup_steps,
#     num_training_steps=(len(train_dataloader) * config.num_epochs),
# )

# def evaluate(config, epoch, pipeline):
#     # 从随机噪声中生成一些图像（这是反向扩散过程）。
#     # 默认的管道输出类型是 `List[torch.Tensor]`
#     # 取sample_channel来展示
#     # print("生成图像......")

#     images = pipeline(
#         batch_size=config.eval_batch_size,
#         generator=torch.Generator(device='cpu').manual_seed(config.seed),  # 使用单独的 torch 生成器来避免回绕主训练循环的随机状态
#         output_type="np.array",
#     ).images

#     print("生成图像维度：", images.shape)
#     # (batch, height, width, channel)

#     # 将生成的eval_batch_size个图像拼接成一张大图
#     fig, ax = plt.subplots(2, 10, figsize=(20, 4))
#     for i in range(2):
#         for j in range(10):
#             ax[i, j].imshow(images[i * 10 + j, :, :, sample_channel], aspect='auto')
#             ax[i, j].axis("off")
#             ax[i, j].set_title(f"Image {i * 10 + j}")

#     plt.savefig(f"figs/{epoch:04d}.png", dpi=400)
#     plt.close()


# def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
#     # Initialize accelerator and tensorboard logging
#     accelerator = Accelerator(
#         mixed_precision=config.mixed_precision,
#         gradient_accumulation_steps=config.gradient_accumulation_steps,
#         log_with="tensorboard",
#         project_dir=os.path.join(config.output_dir, "logs"),
#     )
#     if accelerator.is_main_process:
#         if config.output_dir is not None:
#             os.makedirs(config.output_dir, exist_ok=True)
#         if config.push_to_hub:
#             repo_id = create_repo(
#                 repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
#             ).repo_id
#         accelerator.init_trackers("train_example")

#     # Prepare everything
#     # There is no specific order to remember, you just need to unpack the
#     # objects in the same order you gave them to the prepare method.
#     model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
#         model, optimizer, train_dataloader, lr_scheduler
#     )

#     global_step = 0

#     # Now you train the model
#     for epoch in range(config.num_epochs):
#         progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
#         progress_bar.set_description(f"Epoch {epoch}")

#         for step, batch in enumerate(train_dataloader):
#             clean_images = batch
#             # Sample noise to add to the images
#             noise = torch.randn(clean_images.shape, device=clean_images.device)
#             bs = clean_images.shape[0]

#             # Sample a random timestep for each image
#             timesteps = torch.randint(
#                 0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
#                 dtype=torch.int64
#             )

#             # Add noise to the clean images according to the noise magnitude at each timestep
#             # (this is the forward diffusion process)
#             noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

#             with accelerator.accumulate(model):
#                 # Predict the noise residual
#                 noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
#                 loss = F.mse_loss(noise_pred, noise)
#                 accelerator.backward(loss)

#                 if accelerator.sync_gradients:
#                     accelerator.clip_grad_norm_(model.parameters(), 1.0)
#                 optimizer.step()
#                 lr_scheduler.step()
#                 optimizer.zero_grad()

#             progress_bar.update(1)
#             logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
#             progress_bar.set_postfix(**logs)
#             accelerator.log(logs, step=global_step)
#             global_step += 1

#         # After each epoch you optionally sample some demo images with evaluate() and save the model
#         if accelerator.is_main_process:
#             pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

#             if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
#                 evaluate(config, epoch, pipeline)

#             if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
#                 if config.push_to_hub:
#                     upload_folder(
#                         repo_id=repo_id,
#                         folder_path=config.output_dir,
#                         commit_message=f"Epoch {epoch}",
#                         ignore_patterns=["step_*", "epoch_*"],
#                     )
#                 else:
#                     pipeline.save_pretrained(config.output_dir)
                    

# # --------------------------------------------------
# # Start Train
# # --------------------------------------------------
# from accelerate import notebook_launcher

# args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

# notebook_launcher(train_loop, args, num_processes=1)

