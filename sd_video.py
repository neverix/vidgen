from argparse import Namespace
get_ipython = lambda: Namespace(system= lambda *args: **kwargs: None)
#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -U diffusers transformers')


# In[2]:


# !apt install mpich -y


# In[ ]:


# !conda install -y mpi4py -c conda-forge
get_ipython().system('conda install -y -c conda-forge openmpi=4.1.4')
get_ipython().system('conda install -y -c conda-forge mpi4py')
from mpi4py import MPI
import mpi4py
get_ipython().system('pip install deepspeed')
from pydantic import BaseModel
BaseModel.Config.arbitrary_types_allowed = True
import deepspeed


# In[ ]:


import os
os.environ["JAX_PLATFORMS"] = ""

import torch


torch.set_grad_enabled(True)

from diffusers import UNet2DConditionModel
import torch


device = "cpu"


def get_unet(device=device, is_fp16=True):
    if not is_fp16:
        raise NotImplementedError
    return UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4",
                                                 torch_dtype=torch.float16,  # torch.float32,
                                                 in_channels=8,  # revision="fp16",
                                                 subfolder="unet",
                                           ).to(device)



from functools import partial
from torch import nn
import copy


class VideoAttn(nn.Module):
    def __init__(self, x, window=3):
        super().__init__()
        self.window = window
        self.att1 = x
        self.att2 = copy.deepcopy(x)
        torch.nn.init.zeros_(self.att2.proj_out.weight)
        torch.nn.init.zeros_(self.att2.proj_out.bias)
    
    def forward(self, x, context=None):
        time, context = context
        x = self.att1(x, context=context)  # .repeat(len(x) // len(context), 1, 1)
        x_ = x.reshape(x.shape[0] // self.window, self.window, x.shape[1], -1)
        x_ = x_.permute(0, 3, 2, 1)
        ctx = context[::len(context) // len(x_)]
        z = x_.reshape(-1, *x_.shape[2:]).unsqueeze(-1)
        
        def _attention(self, time, query, key, value):
            # Calculate weightings
            ms = torch.linspace(1, 8, query.shape[0] // self.heads).repeat(self.heads)
            ms = 1 / (torch.ones_like(ms) * 2).pow(ms)
            # TODO: use baddbmm for better performance
            attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
            # ALiBi, assuming self-attention
            x, y = torch.meshgrid(*([
#                 torch.arange(query.shape[1])
                time.detach().cpu().long()
            ] * 2))
            sub = ((x - y).abs()).unsqueeze(0).repeat(query.shape[0], 1, 1) * ms.unsqueeze(-1).unsqueeze(-1)
            attention_probs = attention_scores.sub(sub.to(attention_scores)).softmax(dim=-1)
            # compute attention output
            hidden_states = torch.matmul(attention_probs, value)
            # reshape hidden_states
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
            return hidden_states

        for block in self.att2.transformer_blocks:
            at = block.attn1
            at._attention = partial(_attention, at, time)
        y = self.att2(z, context=ctx.repeat(z.shape[0] // ctx.shape[0], 1, 1))
        y = y[:, :, :, 0].reshape(x_.shape).permute(0, 3, 2, 1).reshape(x.shape)
        return y


class VideoConv(nn.Module):
    def __init__(self, x, window=3):
        super().__init__()
        self.in_channels = x.in_channels
        self.out_channels = x.out_channels
        self.window = window
        self.conv1 = x
        self.conv2 = nn.Conv1d(x.in_channels, x.out_channels, 3, padding="same")
        torch.nn.init.zeros_(self.conv2.weight)
        torch.nn.init.zeros_(self.conv2.bias)
    
    def forward(self, x):
        o = self.conv1(x)
        x_ = x.reshape(x.shape[0] // self.window, self.window, x.shape[1], -1)
        x_ = x_.permute(0, 3, 2, 1)
        z = x_.reshape(-1, *x_.shape[2:])
        y = self.conv2(z)
        y = y.reshape(x_.shape[0], x_.shape[1], self.out_channels, *x_.shape[3:]
                     ).permute(0, 3, 2, 1).reshape(x.shape[0], self.out_channels, *x.shape[2:])
        return o + y


# In[ ]:


unet = get_unet()


# In[ ]:


from diffusers.models.attention import Transformer2DModel


win = 6


def replace(x, k, window=win):
    y = getattr(x, k)
    if isinstance(y, Transformer2DModel):
        z = VideoAttn(y, window)
    elif "VideoAttn" in str(type(y)):
        z = VideoAttn(y.att1, window)
    elif isinstance(y, nn.Conv2d):
        z = VideoConv(y, window)
    elif "VideoConv" in str(type(y)):
        z = VideoConv(y.conv1, window)
    setattr(x, k, z)
    return z


from tqdm.auto import tqdm
import itertools


def fixinate(unet, window=2):
    replace(unet, "conv_in", window)
    replace(unet, "conv_out", window)
    for block in tqdm(list(itertools.chain(unet.down_blocks, [unet.mid_block], unet.up_blocks))):
        if hasattr(block, "resnets"):
            for r in block.resnets:
                replace(r, "conv1", window)
                replace(r, "conv2", window)
        if hasattr(block, "attentions"):
            n_blocks = len(block.attentions)
            for i in range(len(block.attentions)):
                replace(block.attentions, str(i), window)


block_size = 8
fixinate(unet, block_size)


# In[ ]:


get_ipython().system('wget -c https://huggingface.co/datasets/nev/anime-giph/resolve/main/vids.zip')


# In[ ]:


get_ipython().system('unzip -n vids.zip > /dev/null')


# In[ ]:


get_ipython().system('wget -c https://huggingface.co/datasets/nev/anime-giph/resolve/main/data.json')


# In[ ]:

import zipfile
zf = zipfile.ZipFile("vids.zip")
# import glob
vids = [name for name in archive.namelist() if name.endswith(".mp4")]  # glob.glob("vids2/**/*.mp4", recursive=True)
import json
names = {sample["id"]: ",".join(sample["tags"]) for sample in json.load(open("data.json"))}

import os
# sorry non-mp4 users
names = [names.get(os.path.split(os.path.split(vid)[0])[1], os.path.basename(vid)[:-4]) for vid in vids]
print(names[:10])


# In[13]:


import imageio as iio
import numpy as np
import random
import cv2


def load_vid(path, block_size=4):
    data = iio.mimread(zf.open(path, "rb"), "mp4")
    # cap = cv2.VideoCapture(path)
    # n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    n_frames = len(data)
    observable = {}
    blocks = set()
    while len(blocks) < block_size:
        n_group = random.randint(1, block_size)
        frame_skip = np.exp(random.random() * np.log(n_group))
        start = random.random() * (n_frames - frame_skip * (n_group - 1))
        is_observable = random.random() > 0.5
        new_frames = set(int(start + i * frame_skip) for i in range(n_group)) - blocks
        blocks.update(new_frames)
        for frame in sorted(new_frames):
            observable[frame] = is_observable
    else:
        frame_skip = 1
    blocks = sorted(blocks)[:block_size]
    # start = min(blocks)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    obs = []
    for i, frame in enumerate(data):  # range(int(n_frames)):
        # ret, frame = cap.read()
        # if frame is None:
        #     frame = frames[-1]
        if i in blocks:
            frames.append(frame)  # cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            obs.append(observable[i])
    cap.release()
    return np.asarray(frames), np.asarray(blocks), np.asarray(obs)

# vid, fps = load_vid(vids[0])
# print(vid[:2, :2, :2], fps)


# In[14]:


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, names, paths, block_size=4):
        assert len(names) == len(paths)  # This should be blocking
        self.names = names
        self.paths = paths
        self.block_size = block_size
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, i):
        vid, pos, observable = load_vid(self.paths[i], block_size=self.block_size)
        return self.names[i], torch.from_numpy(vid), pos, observable


vds = VideoDataset(names, vids, block_size=block_size)
print(len(vds), vds[0][0], vds[0][1][:1, :2, :2])


from diffusers import AutoencoderKL, DDIMScheduler
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
ddim = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")


from transformers import CLIPTextModel, CLIPTokenizer
text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder")
tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")


from torch.utils.data import Dataset, DataLoader


class VideoProcessor(Dataset):
    def __init__(self, ds, vae, scheduler, tokenizer, text_encoder, target=(256, 128)):
        super().__init__()
        self.ds = ds
        self.vae = vae
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.target_w, self.target_h = target
    
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        text, video, pos, observable = self.ds[i]
        toks = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        text = self.text_encoder(toks)[0]
        
        _, h, w, _ = video.shape
        scale = max(w / self.target_w, h / self.target_h)
        video = torch.nn.functional.interpolate(video.permute(0, 3, 1, 2), scale_factor=1 / scale)
        w_left = self.target_w - video.shape[-1]
        w_compensate = w_left // 2
        h_left = self.target_h - video.shape[-2]
        h_compensate = h_left // 2
        video = torch.nn.functional.pad(video, (w_compensate, w_left - w_compensate, h_compensate, h_left - h_compensate))
        video = video / 127.5 - 1.
        video = self.vae.encode(video).latent_dist.sample() * 0.18215
        
        noise = torch.randn_like(video)
        
        ts = torch.LongTensor(self.scheduler.timesteps)
        t = torch.randint(0, len(ts), (1,)).long()
        t = ts[t]
        
        alpha = self.scheduler.alphas_cumprod[t].item()
        v = (alpha ** 0.5) * video + ((1 - alpha) ** 0.5) * noise
        obs_array = torch.from_numpy(observable).float()
        weight = (1 - obs_array) * (alpha / (1 - alpha))
        
        noised = self.scheduler.add_noise(video, noise, t)
        noised[observable] = video[observable]
        inputs = torch.cat((noised,
                            noised[:, :1] * 0
                                + obs_array[:, None, None, None].repeat(1, *noised.shape[1:])), dim=1)
        text = text.repeat(video.shape[0], 1, 1)
        return (inputs, torch.from_numpy(pos).float(), t, text), v, weight

        
ds = VideoProcessor(vds, vae, ddim, tokenizer, text_encoder)
ds[0][0][-1][0], ds[0][1][:2, :2, :2, :2]


from torch.utils.data import random_split
train_split = int(len(ds) * 0.8)
train_ds, test_ds = random_split(ds, [train_split, len(ds) - train_split])


bs = 1
dl = DataLoader(ds, batch_size=bs)
train_dl = DataLoader(train_ds, batch_size=bs)
test_dl = DataLoader(test_ds, batch_size=bs)


lr = 5e-5  #@param {type: "number"}
wd = 1e-5  #@param {type: "number"}
adam_kwargs = dict(lr=lr, weight_decay=wd, betas=(0.9, 0.99), eps=1e-8)
params = unet.parameters()

from transformers import get_cosine_schedule_with_warmup
from argparse import Namespace
import deepspeed
_, optimizer, *_ = deepspeed.initialize(args=Namespace(**{}), model=unet, model_parameters=params, config={
    "optimizer": {
        "type": "Adam",
        "params": adam_kwargs
    },
    "offload_optimizer": "cpu",
    "train_batch_size": bs
})
# optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)
epochs = 4
lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 500, len(train_dl) * epochs)

from accelerate import Accelerator
accelerator = Accelerator(gradient_accumulation_steps=4)
unet, optimizer, train_dl, scheduler = accelerator.prepare(
    unet, optimizer, train_dl, lr_scheduler
)
unet.requires_grad_(True)
unet.enable_gradient_checkpointing()
unet.step = (lambda x, pos, t, text: unet(x.reshape(-1, *x.shape[-3:]),
                                     t.repeat(1, x.shape[1]).flatten(),
                                     (pos, text.reshape(-1, *text.shape[-2:]))).sample.reshape(x.shape))
device = accelerator.device
use_autocast = False
from PIL import Image
from tqdm.auto import tqdm
def sample():
    with torch.no_grad(), torch.autocast("cuda", enabled=use_autocast):
        ddim.set_timesteps(50)
        noise = torch.randn(block_size, 4, 16, 32).to(device)
        toks = tokenizer(
                "anime based fight",
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids
        text = text_encoder(toks)[0].repeat(block_size, 1, 1)
        for i in tqdm(ddim.timesteps):
            noise = ddim.step(unet(noise, i.to(device), text.to(device)).sample, i.to(device), noise).prev_sample
        decoded = vae.decode(noise.cpu() / 0.18215).sample
        imgs = ((decoded / 2 + 0.5).detach().cpu().clamp(0., 1.).numpy() * 255).astype("uint8")
        for i, img in enumerate(imgs):
            Image.fromarray(img.transpose(1, 2, 0)).save(f"imgs/{i}.png")


from tqdm.auto import tqdm
for _ in range(epochs):
    for i, sample in enumerate(tqdm(train_dl)):
        with accelerator.accumulate(unet), torch.autocast("cuda", enabled=use_autocast):
            optimizer.zero_grad()
            # TODO add position condition
            noise = unet.step(*(k.to(device) for k in sample[0]))
            loss = torch.nn.functional.mse_loss(noise, sample[1]) * sample[2]
            accelerator.backward(loss)
            accelerator.log({"train_loss": loss}, step=1)
        if i % 500 == -1:
            accelerator.save_state("checkpoints/")
            sample()
accelerator.end_training()


# import pytorch_lightning as pl


#class UNetModule(pl.LightningModule):
    #def __init__(self, unet, window=8):
        #super().__init__()
        #self.window = window
        #self.unet = unet
        #fixinate(self.unet, window=window)
        #self.unet.enable_gradient_checkpointing()
        #self.unet.requires_grad_(True)
    

    #def forward(self, x, t, text):
        #return self.unet(noised.reshape(-1, *noised.shape[-3:]),
                         #t.repeat(1, noised.shape[1]).flatten(),
                         #text.reshape(-1, *text.shape[-2:])).sample


    #def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.unet.parameters(), lr=1e-4)
        #epochs = 4
        #scheduler = get_cosine_schedule_with_warmup(optimizer, 500, len(dl) * epochs)
        #return [optimizer], [scheduler]
    

    #def training_step(self, batch, batch_idx):
        #(noised, t, text), noise = batch
        #pred = self(noised, t, text)
        #loss = torch.nn.functional.mse_loss(pred, noise)
        #self.log("loss", loss)
        #return loss


#model = UNetModule(get_unet())
#!wandb login  # !pip install wandb
#trainer = pl.Trainer(
    #precision=16,
    #logger=pl.loggers.WandbLogger(project="smol-stable-diffusion"),
    #default_root_dir="/kaggle/working"
#)
#trainer.fit(model, train_dl, test_dl)
