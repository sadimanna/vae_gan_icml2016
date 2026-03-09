import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms


def build_dataset(opt):
    if opt.dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = dset.ImageFolder(
            root=opt.dataroot,
            transform=transforms.Compose([
                transforms.Resize(opt.imageSize),
                transforms.CenterCrop(opt.imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
    elif opt.dataset == 'lsun':
        dataset = dset.LSUN(
            db_path=opt.dataroot,
            classes=['bedroom_train'],
            transform=transforms.Compose([
                transforms.Resize(opt.imageSize),
                transforms.CenterCrop(opt.imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
    elif opt.dataset == 'cifar10':
        dataset = dset.CIFAR10(
            root=opt.dataroot,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(opt.imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
    elif opt.dataset == 'mnist':
        dataset = dset.MNIST(
            root=opt.dataroot,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(opt.imageSize),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
    else:
        raise NotImplementedError('Dataset not supported: %s' % opt.dataset)

    return dataset


def build_dataloader(opt):
    dataset = build_dataset(opt)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers),
    )
