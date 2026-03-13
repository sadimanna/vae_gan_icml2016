import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
    _TORCHMETRICS_AVAILABLE = True
except Exception:
    FrechetInceptionDistance = None
    LearnedPerceptualImagePatchSimilarity = None
    StructuralSimilarityIndexMeasure = None
    _TORCHMETRICS_AVAILABLE = False


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

        self.device = getattr(
            opt,
            'device',
            torch.device('cuda' if opt.cuda and torch.cuda.is_available() else 'cpu'),
        )

        self.input_x = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize).to(self.device)
        self.noise = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1).to(self.device)
        self.fixed_noise = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1).normal_(0, 1).to(self.device)
        self.label = torch.FloatTensor(opt.batchSize).to(self.device)
        self.real_label = 1
        self.fake_label = 0

        if self.device.type != 'cpu':
            self.netD.to(self.device)
            self.netG.to(self.device)
            self.encoder.to(self.device)
            self.sampler.to(self.device)
            self.criterion.to(self.device)
            self.mse_criterion.to(self.device)

        self.input_x = Variable(self.input_x)
        self.label = Variable(self.label)
        self.noise = Variable(self.noise)
        self.fixed_noise = Variable(self.fixed_noise)

        # setup optimizers: separate for encoder and generator
        # self.optimizerD = optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        # self.optimizerEnc = optim.Adam(self.encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        # self.optimizerG = optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerD = optim.RMSprop(self.netD.parameters(), lr=opt.lr)
        self.optimizerG = optim.RMSprop(self.netG.parameters(), lr=opt.lr)
        self.optimizerEnc = optim.RMSprop(self.encoder.parameters(), lr=opt.lr)

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
    
    def _set_requires_grad(self, model, requires_grad):
        for param in model.parameters():
            param.requires_grad = requires_grad

    def _recon_loss(self, rec, rec_feats):
        if rec_feats:
            with torch.no_grad():
                _, real_feats = self.netD(self.input_x)
            if not real_feats:
                return self.mse_criterion(rec, self.input_x)
            return self._feature_mse(rec_feats, real_feats)
        return self.mse_criterion(rec, self.input_x)

    def _normalize_to_01(self, tensor, eps=1e-8):
        # Normalize tensor to [0, 1] range for image saving.
        t_min = tensor.amin()
        t_max = tensor.amax()
        return (tensor - t_min) / (t_max - t_min + eps)

    def _denorm_to_01(self, tensor):
        # Data pipeline normalizes to [-1, 1]. Map back to [0, 1] for metrics.
        return ((tensor + 1.0) / 2.0).clamp(0.0, 1.0)

    def _build_metrics(self):
        if not _TORCHMETRICS_AVAILABLE:
            return None
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(self.device)
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(self.device)
        return fid, ssim, lpips

    def evaluate(self, dataloader):
        opt = self.opt
        eval_samples = int(getattr(opt, 'eval_samples', 0))
        if eval_samples <= 0:
            self.logger.info('Skipping evaluation: eval_samples <= 0')
            return

        metrics = self._build_metrics()
        if metrics is None:
            self.logger.warning('Skipping evaluation: torchmetrics not available')
            return

        fid, ssim, lpips = metrics

        self.netG.eval()
        self.encoder.eval()
        self.sampler.eval()

        seen = 0
        with torch.no_grad():
            for real_cpu, _ in dataloader:
                if seen >= eval_samples:
                    break
                real = real_cpu.to(self.device)
                batch_size = real.size(0)
                remaining = eval_samples - seen
                if remaining < batch_size:
                    real = real[:remaining]
                    batch_size = real.size(0)

                real_01 = self._denorm_to_01(real)
                fid.update(real_01, real=True)

                noise = torch.randn(batch_size, opt.nz, 1, 1, device=self.device)
                fake = self.netG(noise)
                fake_01 = self._denorm_to_01(fake)
                fid.update(fake_01, real=False)

                encoded = self.encoder(real)
                sampled = self.sampler(encoded)
                rec = self.netG(sampled)
                rec_01 = self._denorm_to_01(rec)

                ssim.update(rec_01, real_01)
                lpips.update(rec_01, real_01)

                seen += batch_size

        fid_score = fid.compute().item()
        ssim_score = ssim.compute().item()
        lpips_score = lpips.compute().item()

        self.logger.info(
            'Eval metrics over %d samples | FID: %.4f | SSIM: %.4f | LPIPS: %.4f',
            seen,
            fid_score,
            ssim_score,
            lpips_score,
        )

    def train(self, dataloader):
        opt = self.opt
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(opt.niter):
            for i, data in enumerate(dataloader, 0):
                self.encoder.zero_grad()
                self.netD.zero_grad()
                self.netG.zero_grad()
                ############################
                real_cpu, _ = data
                real = real_cpu.to(self.device)
                batch_size = real.size(0)
                with torch.no_grad():
                    self.input_x.resize_(real.size()).copy_(real)
                ############################
                # (1) Encoder first
                ###########################
                x_enc = self.encoder(self.input_x)
                mu = x_enc[0]
                logvar = x_enc[1]
                if not torch.isfinite(mu).all() or not torch.isfinite(logvar).all():
                    self.logger.warning('Skipping batch due to non-finite encoder outputs')
                    continue
                logvar = torch.clamp(logvar, min=-10.0, max=10.0)
                x_enc = [mu, logvar]
                ############################
                # (2) Generate samples from noise, and reconstructed samples from encoded real data
                ###########################
                sampled = self.sampler(x_enc)
                rec = self.netG(sampled) # ------------------------> Reconstruction
                ############################
                # (3) Pass through discriminator
                rec_output, rec_feats = self.netD(rec)
                ############################
                # (4) calculate loss for encoder
                ############################
                KLD_element = mu.pow(2) + logvar.exp() - 1 - logvar
                KLD_loss = -0.5 * torch.sum(KLD_element)
                Disllike_loss = self._recon_loss(rec, rec_feats)
                VAEerr = opt.kld_wt * KLD_loss + Disllike_loss
                VAEerr.backward()
                self.optimizerEnc.step()
                ############################
                # (5) Calculate loss for Generator / Decoder
                ###########################
                self.netG.zero_grad()
                self.netD.zero_grad()

                sampled_detached = sampled.detach()
                rec = self.netG(sampled_detached)
                self._set_requires_grad(self.netD, False)
                rec_output, rec_feats = self.netD(rec)

                with torch.no_grad():
                    self.noise.resize_(batch_size, opt.nz, 1, 1)
                    self.noise.normal_(0, 1)
                gen = self.netG(self.noise)
                gen_output, _ = self.netD(gen)

                label_fake = torch.full_like(gen_output, float(self.fake_label))
                bce_fake = self.criterion(gen_output, label_fake)
                label_rec = torch.full_like(rec_output, float(self.fake_label))
                bce_rec = self.criterion(rec_output, label_rec)
                Disllike_loss = self._recon_loss(rec, rec_feats)
                Decerr = - (bce_fake + bce_rec) + opt.gamma * Disllike_loss # (1 - opt.gamma) *
                Decerr.backward()
                self.optimizerG.step()
                #################################
                # (6) Calculate Loss for Discriminator
                ###########################
                self._set_requires_grad(self.netD, True)
                self.netD.zero_grad()
                output, _ = self.netD(self.input_x)
                label_real = torch.full_like(output, float(self.real_label))
                bce_real = self.criterion(output, label_real)
                D_x = output.data.mean()
                output_fake, _ = self.netD(gen.detach())
                label_fake = torch.full_like(output_fake, float(self.fake_label))
                bce_fake = self.criterion(output_fake, label_fake)
                D_G_z1 = output_fake.data.mean()
                output_rec, _ = self.netD(rec.detach())
                label_rec = torch.full_like(output_rec, float(self.fake_label))
                bce_rec = self.criterion(output_rec, label_rec)
                D_G_z_rec = output_rec.data.mean()
                Diserr = bce_real + bce_fake + bce_rec
                Diserr.backward()
                self.optimizerD.step()
                #################################

                if i % 500 == 0:
                    # save generated batch to file
                    vutils.save_image(
                        self._normalize_to_01(gen),
                        os.path.join(opt.outf, 'gen_epoch%d_iter%d.png' % (epoch, i)),
                        normalize=False,
                    )
                    # save reconstruction batch to file
                    vutils.save_image(
                        self._normalize_to_01(rec),
                        os.path.join(opt.outf, 'rec_epoch%d_iter%d.png' % (epoch, i)),
                        normalize=False,
                    )

                    # print(f"Range of outputs: {gen.min().item()}-{gen.max().item()}")
                    # gen.clamp_(min=-1.0, max=1.0)
                    # gen.sub_(-1.0).div_(max(2.0, 1e-5))
                    # print(f"Range of outputs: {gen.min().item()}-{gen.max().item()}")

                    self.logger.info(
                        '[%d/%d][%d/%d] Loss_VAE: %.4f Loss_D: %.4f Loss_G: %.4f '
                        'D(x): %.4f D(G(z)): %.4f D(Dec(Enc(x))): %.4f',
                        epoch + 1,
                        opt.niter,
                        i,
                        len(dataloader),
                        VAEerr.item(),
                        Diserr.item(),
                        Decerr.item(),
                        D_x,
                        D_G_z1,
                        D_G_z_rec,
                    )

                    if self.prevDLoss is not None and self.prevGLoss is not None:
                        if Diserr.item() > self.prevDLoss and abs(Decerr.item()) > self.prevGLoss:
                            self.patience -= 1
                            self.logger.info('No improvement in D or G loss, reducing patience to %d', self.patience)
                            if self.patience <= 0:
                                self.logger.info('Early stopping at epoch %d, iteration %d', epoch + 1, i)
                                break
                        else:
                            self.patience = opt.stopIter
                    self.prevDLoss = Diserr.item()
                    self.prevGLoss = abs(Decerr.item())

            if self.patience <= 0:
                self.logger.info('Early stopping at epoch %d, iteration %d', epoch + 1, i)
                break

            if (epoch + 1) % opt.saveInt == 0 and epoch != 0:
                torch.save(self.netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch + 1))
                torch.save(self.netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch + 1))
                torch.save(self.encoder.state_dict(), '%s/encoder_epoch_%d.pth' % (opt.outf, epoch + 1))

        self.evaluate(dataloader)
