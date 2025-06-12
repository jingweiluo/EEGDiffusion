import torch
import numpy as np
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
# from denoising_diffusion_pytorch.utils import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Unet(
    dim = 8, # embed_init_dim
    dim_mults = (1, 2, 4, 8),
    channels=5 # 初始通道数
)

diffusion = GaussianDiffusion(
    model,
    image_size = 64,
    timesteps = 1000,   # number of steps
    # loss_type = 'l1'    # L1 or L2
).to(device)

# training_images = torch.randn(8, 5, 64, 64).to(device) # images are normalized from 0 to 1

test_data = np.load('data/tf_data.npz')
training_images = test_data['X_tf']
training_images  = np.abs(training_images).astype(np.float32)
y = test_data['y']
training_images = torch.from_numpy(training_images).float().to(device)
training_images= training_images[:400]


print(type(training_images), training_images.shape)



# trainer = Trainer(
#     diffusion,
#     'data/stft.npy',
#     train_batch_size=8,
#     train_lr=8e-5,
#     train_num_steps=700000,  # total training steps
#     gradient_accumulate_every=2,  # gradient accumulation steps
#     ema_decay=0.995,  # exponential moving average decay
#     amp=True  # turn on mixed precision
# )

# trainer.train()



loss = diffusion(training_images)
loss.backward()

sampled_images = diffusion.sample(batch_size = 8)
sampled_images = sampled_images.detach().cpu().numpy()
np.save(f"generated_data/Tosato_generated_eeg_stft.npy", sampled_images)
print(f"已保存")

# import torch
# from denoising_diffusion_pytorch import Unet, GaussianDiffusion

# model = Unet(
#     dim = 5,
#     dim_mults = (1, 2, 4, 8),
#     channels=5,

# )

# diffusion = GaussianDiffusion(
#     model,
#     image_size = 64,
#     timesteps = 1000,   # number of steps
#     # loss_type = 'l1'    # L1 or L2
# )

# training_images = torch.randn(8, 5, 64, 64) # images are normalized from 0 to 1
# loss = diffusion(training_images)
# loss.backward()
# # after a lot of training

# sampled_images = diffusion.sample(batch_size = 4)
# sampled_images.shape # (4, 3, 128, 128)