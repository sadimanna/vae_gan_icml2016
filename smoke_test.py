import sys
import argparse
import traceback
try:
    import torch
    import main as m
except Exception:
    traceback.print_exc()
    raise
# set required globals used by classes
m.ngf = 64
m.nz = 100
m.nc = 3
m.ndf = 64
# ensure opt exists for sampler
m.opt = argparse.Namespace(cuda=False)
# instantiate models
enc = m._Encoder(64)
samp = m._Sampler()
netG = m._netG(m.nz, m.ngf, m.nc, ngpu=1)
# discriminator imageSize should match generator output (32 for this DCGAN variant)
netD = m._netD(32, ngpu=1)
# dummy inputs
x = torch.randn(2, m.nc, 64, 64)
noise = torch.randn(2, m.nz, 1, 1)
# forward passes
enc_out = enc(x)
print('enc mu shape', enc_out[0].shape, 'enc logvar shape', enc_out[1].shape)
sampled = samp(enc_out)
print('sampled shape', sampled.shape)
rec = netG(sampled)
print('rec shape', rec.shape)
# step through discriminator modules to find shape progression
x_in = rec
for idx, module in enumerate(netD.main):
    x_in = module(x_in)
    print('after module', idx, module.__class__.__name__, 'shape', x_in.shape)
out = x_in
print('disc out shape', out.shape)
print('smoke test completed')
