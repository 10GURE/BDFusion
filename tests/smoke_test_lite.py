import torch, time
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

config = OmegaConf.load('./informationSynthesis/Image2Image_Lite.yaml')
model = instantiate_from_config(config.model)
state = torch.load('./checkpoint/last.ckpt', map_location='cpu')['state_dict']
for k, p in model.state_dict().items():
    if k.startswith('first_stage_model.') and k in state and state[k].shape == p.shape:
        p.copy_(state[k])
model = model.cuda().eval()

feats = model.model.encoder_v(torch.randn(1, 3, 64, 64).cuda())
print('Encoder:', [f.shape[1] for f in feats])
print('UNet ch:', [model.model.diffusion_model.model_channels * cm 
      for cm in model.model.diffusion_model.channel_mult])

c1 = torch.randn(1, 3, 64, 64).cuda()
c2 = torch.randn(1, 3, 64, 64).cuda()
x = torch.randn(1, 3, 64, 64).cuda()
o = model.apply_model(x, torch.tensor([500], device='cuda').long(), c1, c2)
print(f'apply_model: {list(x.shape)} -> {list(o.shape)} OK')

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    model.apply_model(torch.randn(1, 3, 64, 64).cuda(), 
                      torch.tensor([500], device='cuda').long(), c1, c2)
torch.cuda.synchronize()
print(f'UNet step: {(time.perf_counter()-t0)/10*1000:.1f} ms (orig ~56ms)')

t = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f'Params: {t/1e6:.1f}M train / {total/1e6:.1f}M total')
