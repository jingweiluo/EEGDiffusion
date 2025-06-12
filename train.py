from read_example import plot_tf_trials
# --------------------------------------------------
# Define Training Config
# --------------------------------------------------
from dataclasses import dataclass
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class TrainingConfig:
    # 类别总数
    num_classes = 23
    # 图像尺寸
    image_size = (64, 64)
    # 训练批次大小
    train_batch_size = 8
    # 评估批次大小
    eval_batch_size = 8
    # 训练轮数
    num_epochs = 60
    # 梯度累积步数（累计几次梯度更新一次参数）
    gradient_accumulation_steps = 1
    # 学习率
    learning_rate = 1e-4
    # 学习率衰减
    lr_warmup_steps = 500
    save_image_epochs = 5
    save_model_epochs = 20
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "results"
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
X_tf = test_data['X_tf']
y = test_data['y']

X_tf = torch.tensor(X_tf, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
print("转换前的维度：", X_tf.shape)

# X_tf_resize = transform_image(X_tf)
# print("转换后的维度：", X_tf_resize.shape)
X_tf_resize = X_tf

# 【optional】此处可以进行norm操作
X_tf_resize_norm = X_tf_resize


# --------------------------------------------------
# Define UNet2DModel
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
    class_embed_type="timestep",        # 或 "projection"
    num_class_embeds=config.num_classes,  # 类别总数，如 23
    dropout = 0.5
)


# --------------------------------------------------
# Training Preparation
# --------------------------------------------------
import os
from pathlib import Path
from tqdm.auto import tqdm
from tqdm import trange

from accelerate import Accelerator
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn.functional as F

from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
from diffusers import DDPMScheduler

from huggingface_hub import create_repo, upload_folder

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
dataset = TensorDataset(X_tf_resize_norm, y)
train_dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

@torch.no_grad()
def evaluate(config, epoch, model_dir):
    """
    手动从 model_dir 加载 UNet 和 Scheduler，并做反向扩散采样。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载 UNet 和 scheduler
    unet = UNet2DModel.from_pretrained(model_dir).to(device)
    scheduler = DDPMScheduler.from_pretrained(model_dir)
    unet.eval()

    # 2. 准备固定标签（示例：全 3 类）
    batch_size = config.eval_batch_size
    class_value = 3
    eval_labels = torch.full(
        (batch_size,),
        fill_value=class_value,
        device=device,
        dtype=torch.int64
    )

    # 3. 随机噪声初始化
    shape = (batch_size, unet.config.in_channels, *unet.config.sample_size)
    imgs = torch.randn(shape, device=device)

    # 4. 反向扩散循环
    for t in trange(len(scheduler.timesteps), desc=f"Sampling epoch {epoch}"):
        timestep = scheduler.timesteps[t]

        # UNet 预测噪声残差
        noise_pred = unet(
            imgs,
            torch.tensor([timestep] * batch_size, device=device, dtype=torch.int64),
            class_labels=eval_labels,
            return_dict=False
        )[0]

        # scheduler 更新样本
        step = scheduler.step(noise_pred, timestep, imgs)
        imgs = step.prev_sample

    # 5. 转 NumPy 并绘图
    images_np = imgs.cpu().permute(0, 2, 3, 1).numpy()   # (B, H, W, C)
    images_tf = images_np.transpose(0, 3, 1, 2)         # (B, C, H, W)
    plot_tf_trials(images_tf, list(range(batch_size)))



def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, (clean_images, labels) in enumerate(train_dataloader):
            clean_images = clean_images.to(device)        # (bs, C, H, W)
            labels = labels.to(device)       # (bs,)
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, class_labels=labels, return_dict=False)[0]
                # loss = F.mse_loss(noise_pred, noise)
                eps = 1e-8
                mse   = F.mse_loss(noise_pred, noise)
                power = noise.pow(2).mean()
                loss  = mse / (power + eps)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # 初始化 best loss
        if epoch == 0:
            best_loss = float("inf")

        # 每个 epoch 训练结束后主进程执行保存和评估
        if accelerator.is_main_process:
            # 1. 解包模型并保存 UNet & scheduler
            unet = accelerator.unwrap_model(model)
            save_dir = f"{config.output_dir}/best_model"
            unet.save_pretrained(save_dir)
            noise_scheduler.save_pretrained(save_dir)

            # 2. 调用 evaluate，传入保存路径
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, save_dir)

            # 3. 根据 avg_loss 决定是否更新最佳模型
            avg_loss = accelerator.gather(torch.tensor(loss.detach().item())).mean().item()
            if avg_loss < best_loss:
                best_loss = avg_loss
                print(f"[Epoch {epoch}] New best loss: {best_loss:.6f} — saving model...")

                # 如果你要推到 Hub，也可以 similarly push save_dir
                if config.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=save_dir,
                        commit_message=f"Best model at epoch {epoch} (loss={best_loss:.6f})",
                        ignore_patterns=["step_*", "epoch_*"],
                    )

# --------------------------------------------------
# Start Train
# --------------------------------------------------
from accelerate import notebook_launcher

args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

notebook_launcher(train_loop, args, num_processes=1)

