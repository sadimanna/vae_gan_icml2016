# VAE/GAN using PyTorch
DCGAN combined with VAE in PyTorch!

This code is based on the paper [Autoencoding beyond pixels using a learned similarity metric](https://proceedings.mlr.press/v48/larsen16.html)


## Requirements
* torch
* torchvision


To start the training:
```
usage: main.py [-h] --dataset DATASET --dataroot DATAROOT [--workers WORKERS]
               [--batchSize BATCHSIZE] [--imageSize IMAGESIZE] [--nz NZ]
               [--ngf NGF] [--ndf NDF] [--niter NITER] [--saveInt SAVEINT] [--lr LR]
               [--beta1 BETA1] [--cuda] [--ngpu NGPU] [--netG NETG]
               [--netD NETD]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     cifar10 | lsun | imagenet | folder | lfw
  --dataroot DATAROOT   path to dataset
  --workers WORKERS     number of data loading workers
  --batchSize BATCHSIZE
                        input batch size
  --imageSize IMAGESIZE
                        the height / width of the input image to network
  --nz NZ               size of the latent z vector
  --ngf NGF
  --ndf NDF
  --niter NITER         number of epochs to train for
  --saveInt SAVEINT     number of epochs between checkpoints
  --lr LR               learning rate, default=0.0002
  --beta1 BETA1         beta1 for adam. default=0.5
  --cuda                enables cuda
  --ngpu NGPU           number of GPUs to use
  --netG NETG           path to netG (to continue training)
  --netD NETD           path to netD (to continue training)
```

## Training Example

```bash
python main.py --dataset cifar10 --dataroot ./data --imageSize 32 --batchSize 64 \
    --niter 100 --cuda --ngpu 1
```

This will create a timestamped run directory under `outputs/` containing:
- Generated image samples (`gen_epoch*.png`, `rec_epoch*.png`)
- Model checkpoints (`netG_epoch_*.pth`, `netD_epoch_*.pth`, `encoder_epoch_*.pth`)
- Training log (`run.log`)

## Generation and Latent Space Visualization

The `generate_and_visualize.py` script loads trained checkpoints and generates samples from the latent space, with t-SNE visualization.

### Usage

```
usage: generate_and_visualize.py [-h] --dataset DATASET --dataroot DATAROOT
                                 [--imageSize IMAGESIZE] [--nz NZ]
                                 [--ngf NGF] [--ndf NDF] [--ngpu NGPU]
                                 [--netG NETG] [--netE NETG]
                                 [--outf OUTF] [--num NUM]
                                 [--perturb_std PERTURB_STD]
                                 [--evalBatch EVALBATCH]
                                 [--cuda]

optional arguments:
  -h, --help                show this help message and exit
  --dataset DATASET         cifar10 | lsun | imagenet | folder | lfw
  --dataroot DATAROOT       path to dataset
  --imageSize IMAGESIZE     height/width of input images (default: 64)
  --nz NZ                   size of latent z vector (default: 100)
  --ngf NGF                 number of feature maps in generator (default: 64)
  --ndf NDF                 number of feature maps in discriminator (default: 64)
  --ngpu NGPU               number of GPUs to use (default: 1)
  --netG NETG               path to generator checkpoint (optional)
  --netE NETE               path to encoder checkpoint (optional)
  --outf OUTF               output folder for visualizations (default: '.')
  --num NUM                 number of samples to generate (default: 200)
  --perturb_std PERTURB_STD std of gaussian perturbation for encoded samples (default: 0.1)
  --evalBatch EVALBATCH     batch size for encoding/generation to avoid OOM (default: 64)
  --cuda                    enable CUDA acceleration
```

### What it does

- **Encodes** a batch of test images using the trained encoder
- **Adds perturbations** to the encoded vectors (Gaussian noise)
- **Generates** images from perturbed latent codes using the generator
- **Produces t-SNE plots** of the 2D-projected latent space, annotated with generated images
- **Saves image grids** for both encoded+perturbed samples and pure random samples

### Example Usage

Generate 1000 samples from a trained model:

```bash
python generate_and_visualize.py \
    --dataset cifar10 \
    --dataroot ./data \
    --imageSize 32 \
    --netG outputs/run_20260303_151218/netG_epoch_100.pth \
    --netE outputs/run_20260303_151218/encoder_epoch_100.pth \
    --outf ./visualizations \
    --num 1000 \
    --perturb_std 0.05 \
    --evalBatch 16 \
    --cuda
```

### Output Files

Created in the `--outf` directory:

- **`tsne_plot.png`** – t-SNE visualization of latent space (10000×10000 px) with embedded generated images
- **`gen_grid.png`** – Grid of images generated from encoded + perturbed test samples
- **`rand_grid.png`** – Grid of images generated from random noise (baseline)

### Notes

- Use `--evalBatch` to control sub-batch size for encoding/generation. Lower values use less GPU memory but are slower.
- The script automatically resizes images to `--imageSize` and applies standard normalization.
- If checkpoint paths are not provided, networks are randomly initialized (useful for testing).
- For large `--num` values (>5000), batch-wise processing prevents out-of-memory errors.