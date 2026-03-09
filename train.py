import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable


class Trainer:
    def __init__(self, opt, encoder, sampler, netG, netD, logger):
        self.opt = opt
        self.encoder = encoder
        self.sampler = sampler
        self.netG = netG
        self.netD = netD
        self.logger = logger

        self.criterion = nn.BCELoss()
        self.mse_criterion = nn.MSELoss()

        self.input_x = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
        self.noise = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1)
        self.fixed_noise = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1).normal_(0, 1)
        self.label = torch.FloatTensor(opt.batchSize)
        self.real_label = 1
        self.fake_label = 0

        if opt.cuda:
            self.netD.cuda()
            self.netG.cuda()
            self.encoder.cuda()
            self.sampler.cuda()
            self.criterion.cuda()
            self.mse_criterion.cuda()
            self.input_x, self.label = self.input_x.cuda(), self.label.cuda()
            self.noise, self.fixed_noise = self.noise.cuda(), self.fixed_noise.cuda()

        self.input_x = Variable(self.input_x)
        self.label = Variable(self.label)
        self.noise = Variable(self.noise)
        self.fixed_noise = Variable(self.fixed_noise)

        # setup optimizers: separate for encoder and generator
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerEnc = optim.Adam(self.encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        self.patience = opt.stopIter
        self.prevDLoss = None
        self.prevGLoss = None

    def _feature_mse(self, feats_a, feats_b):
        losses = []
        for feat_a, feat_b in zip(feats_a, feats_b):
            losses.append(self.mse_criterion(feat_a, feat_b))
        if len(losses) == 1:
            return losses[0]
        return sum(losses) / len(losses)

    def _recon_loss(self, rec, rec_feats):
        if rec_feats:
            with torch.no_grad():
                _, real_feats = self.netD(self.input_x)
            if not real_feats:
                return self.mse_criterion(rec, self.input_x)
            return self._feature_mse(rec_feats, real_feats)
        return self.mse_criterion(rec, self.input_x)

    def train(self, dataloader):
        opt = self.opt
        for epoch in range(opt.niter):
            for i, data in enumerate(dataloader, 0):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(Dec(z))) + log(1-D(Dec(Enc(x))))
                ###########################
                # train with real
                self.netD.zero_grad()
                real_cpu, _ = data
                batch_size = real_cpu.size(0)
                self.input_x.data.resize_(real_cpu.size()).copy_(real_cpu)
                #################################
                # Dis(x)
                self.label.data.resize_(real_cpu.size(0)).fill_(self.real_label)
                output, _ = self.netD(self.input_x)
                errD_real = self.criterion(output, self.label.view(-1, 1))
                D_x = output.data.mean()
                # train with fake - Dis(Dec(z))
                self.noise.data.resize_(batch_size, opt.nz, 1, 1)
                self.noise.data.normal_(0, 1)
                gen = self.netG(self.noise)
                self.label.data.fill_(self.fake_label)
                output, _ = self.netD(gen.detach())
                errD_fake = self.criterion(output, self.label.view(-1, 1))
                D_G_z1 = output.data.mean()
                # train with reconstructed (encoded->sampled->decoded) -- Dis(Dec(Enc(x)))
                encoded = self.encoder(self.input_x)
                sampled = self.sampler(encoded)
                rec = self.netG(sampled)  # ------------------------> Reconstruction
                output, _ = self.netD(rec.detach())
                errD_rec = self.criterion(output, self.label.view(-1, 1))
                D_G_z_rec = output.data.mean()
                errD = errD_real + errD_fake + errD_rec
                errD.backward(retain_graph=False)
                self.optimizerD.step()

                ############################
                # (3) Update G network: maximize log(D(G(z))) + reconstruction loss
                ###########################

                self.label.data.fill_(self.real_label)  # fake labels are real for generator cost
                self.netG.zero_grad()
                # train with fake - Dis(Dec(z))
                self.noise.data.resize_(batch_size, opt.nz, 1, 1)
                self.noise.data.normal_(0, 1)
                gen = self.netG(self.noise)
                ######### Rec part ###########
                self.label.data.fill_(self.fake_label)
                output, _ = self.netD(gen)
                errG_fake = self.criterion(output, self.label.view(-1, 1))
                D_G_z1 = output.data.mean()
                # reconstruct via encoder->sampler->generator
                encoded = self.encoder(self.input_x)
                sampled = self.sampler(encoded)
                rec = self.netG(sampled)
                output, rec_feats = self.netD(rec)
                errG_adv = self.criterion(output, self.label.view(-1, 1))
                # add reconstruction loss to generator training
                if self.netD.hook_layers:
                    _, input_feats = self.netD(self.input_x)
                    MSEerr_G = self._recon_loss(input_feats, rec_feats)
                else:
                    MSEerr_G = self.mse_criterion(rec, self.input_x)
                errG = -errG_fake - errG_adv + opt.gamma * MSEerr_G
                errG.backward(retain_graph=False)
                D_G_z2 = output.data.mean()
                self.optimizerG.step()

                ############################
                # (2) Update Encoder network: minimize KL divergence + reconstruction error
                ###########################

                self.encoder.zero_grad()
                encoded = self.encoder(self.input_x)
                mu = encoded[0]
                logvar = encoded[1]
                KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
                KLD_loss = torch.sum(KLD_element).mul_(-0.5)
                sampled = self.sampler(encoded)
                rec = self.netG(sampled)
                if self.netD.hook_layers:
                    _, rec_feats = self.netD(rec)
                    MSEerr = self._recon_loss(rec, rec_feats)
                else:
                    MSEerr = self.mse_criterion(rec, self.input_x)
                VAEerr = opt.kld_wt * KLD_loss + MSEerr
                VAEerr.backward()
                self.optimizerEnc.step()

                if i % 500 == 0:
                    # save generated batch to file
                    vutils.save_image(
                        gen,
                        os.path.join(opt.outf, 'gen_epoch%d_iter%d.png' % (epoch, i)),
                        normalize=True,
                        value_range=(-1, 1),
                    )
                    # save reconstruction batch to file
                    vutils.save_image(
                        rec,
                        os.path.join(opt.outf, 'rec_epoch%d_iter%d.png' % (epoch, i)),
                        normalize=True,
                        value_range=(-1, 1),
                    )
                    self.logger.info(
                        '[%d/%d][%d/%d] Loss_VAE: %.4f Loss_D: %.4f Loss_G: %.4f '
                        'D(x): %.4f D(G(z)): %.4f D(Dec(Enc(x))): %.4f',
                        epoch + 1,
                        opt.niter,
                        i,
                        len(dataloader),
                        VAEerr.item(),
                        errD.item(),
                        errG.item(),
                        D_x,
                        D_G_z1,
                        D_G_z_rec,
                    )

                    if self.prevDLoss is not None and self.prevGLoss is not None:
                        if errD.item() > self.prevDLoss and abs(errG.item()) > self.prevGLoss:
                            self.patience -= 1
                            self.logger.info('No improvement in D or G loss, reducing patience to %d', self.patience)
                            if self.patience <= 0:
                                self.logger.info('Early stopping at epoch %d, iteration %d', epoch + 1, i)
                                break
                        else:
                            self.patience = opt.stopIter
                            self.prevDLoss = errD.item()
                            self.prevGLoss = abs(errG.item())
                    else:
                        self.prevDLoss = errD.item()
                        self.prevGLoss = abs(errG.item())

            if self.patience <= 0:
                self.logger.info('Early stopping at epoch %d, iteration %d', epoch + 1, i)
                break

            if (epoch + 1) % opt.saveInt == 0 and epoch != 0:
                torch.save(self.netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch + 1))
                torch.save(self.netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch + 1))
                torch.save(self.encoder.state_dict(), '%s/encoder_epoch_%d.pth' % (opt.outf, epoch + 1))
