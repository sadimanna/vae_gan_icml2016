import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
from math import ceil

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

        # setup optimizers: separate for encoder, generator, discriminator
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=opt.lr_dis, betas=(opt.beta1, 0.999))
        self.optimizerEnc = optim.Adam(self.encoder.parameters(), lr=opt.lr_enc, betas=(opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=opt.lr_dec, betas=(opt.beta1, 0.999))
        # self.optimizerD = optim.RMSprop(self.netD.parameters(), lr=opt.lr_dis)
        # self.optimizerG = optim.RMSprop(self.netG.parameters(), lr=opt.lr_dec)
        # self.optimizerEnc = optim.RMSprop(self.encoder.parameters(), lr=opt.lr_enc)

        self.schedulerD = self._build_scheduler(self.optimizerD, opt.lr_decay_dis)
        self.schedulerG = self._build_scheduler(self.optimizerG, opt.lr_decay_dec)
        self.schedulerEnc = self._build_scheduler(self.optimizerEnc, opt.lr_decay_enc)

        self.patience = opt.stopIter
        self.prevDLoss = None
        self.prevGLoss = None
        self.loss_history = {
            "step": [],
            "epoch": [],
            "vae_total": [],
            "kld": [],
            "recon": [],
            "dec_total": [],
            "bce_gen": [],
            "bce_rec_dec": [],
            "dis_total": [],
            "bce_real": [],
            "bce_fake": [],
            "bce_rec_dis": [],
        }

    def _feature_mse(self, feats_a, feats_b):
        losses = []
        for feat_a, feat_b in zip(feats_a, feats_b):
            losses.append(self.mse_criterion(feat_a, feat_b))
        if len(losses) == 1:
            return losses[0]
        return sum(losses) / len(losses)

    def _build_scheduler(self, optimizer, decay):
        name = getattr(self.opt, "scheduler", "exponential").lower()
        total_steps = getattr(self.opt, "scheduler_total_steps", None)
        if total_steps is None:
            total_steps = self.opt.niter
        if name == "none":
            return None
        if name == "exponential":
            return optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay)
        if name == "linear":
            return optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps)
        if name == "step":
            step_size = getattr(self.opt, "scheduler_step_size", None)
            gamma = getattr(self.opt, "scheduler_gamma", 0.9)
            if step_size is None:
                self.logger.info("Defaulting scheduler_step_size to 30%% of total steps")
                step_size = max(1, int(0.3 * total_steps))
            return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        if name == "cosine":
            min_lr = getattr(self.opt, "scheduler_min_lr", 0.0)
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)
        if name in ("cosine_warmup", "cosine-warmup"):
            from utils import CosineWarmupLR
            warmup_steps = getattr(self.opt, "scheduler_warmup_steps", None)
            if warmup_steps is None:
                self.logger.info("Defaulting scheduler_warmup_steps to 10%% of total steps")
                warmup_steps = int(0.1*total_steps)
            min_lr = getattr(self.opt, "scheduler_min_lr", 0.000001)
            return CosineWarmupLR(optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr=min_lr)
        raise ValueError(f"Unknown scheduler: {name}")
    
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
            return {"status": "skipped", "reason": "eval_samples <= 0"}

        metrics = self._build_metrics()
        if metrics is None:
            self.logger.warning('Skipping evaluation: torchmetrics not available')
            return {"status": "skipped", "reason": "torchmetrics not available"}

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
        return {
            "status": "ok",
            "eval_samples": seen,
            "fid": fid_score,
            "ssim": ssim_score,
            "lpips": lpips_score,
        }

    def _step_schedulers(self):
        if self.schedulerEnc is not None:
            self.schedulerEnc.step()
        if self.schedulerG is not None:
            self.schedulerG.step()
        if self.schedulerD is not None:
            self.schedulerD.step()
        lrs = (
            self.optimizerEnc.param_groups[0]["lr"],
            self.optimizerG.param_groups[0]["lr"],
            self.optimizerD.param_groups[0]["lr"],
        )
        self.logger.info(
            "LRs after decay | enc: %.6g | dec: %.6g | dis: %.6g",
            lrs[0],
            lrs[1],
            lrs[2],
        )

    def _record_losses(
        self,
        epoch,
        step,
        vae_total,
        kld,
        recon,
        dec_total,
        bce_gen,
        bce_rec_dec,
        dis_total,
        bce_real,
        bce_fake,
        bce_rec_dis,
    ):
        self.loss_history["step"].append(step)
        self.loss_history["epoch"].append(epoch)
        self.loss_history["vae_total"].append(float(vae_total))
        self.loss_history["kld"].append(float(kld))
        self.loss_history["recon"].append(float(recon))
        self.loss_history["dec_total"].append(float(dec_total))
        self.loss_history["bce_gen"].append(float(bce_gen))
        self.loss_history["bce_rec_dec"].append(float(bce_rec_dec))
        self.loss_history["dis_total"].append(float(dis_total))
        self.loss_history["bce_real"].append(float(bce_real))
        self.loss_history["bce_fake"].append(float(bce_fake))
        self.loss_history["bce_rec_dis"].append(float(bce_rec_dis))

    def _plot_losses(self):
        if not self.loss_history["step"]:
            self.logger.warning("No loss history to plot.")
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            self.logger.warning("Skipping loss plots: matplotlib not available (%s)", exc)
            return

        try:
            import seaborn as sns
            sns.set_theme(style="whitegrid")
        except Exception:
            pass

        labels = {
            "vae_total": "VAE Total (kld_wt*KLD + recon)",
            "kld": "KLD Loss",
            "recon": "Reconstruction Loss",
            "dec_total": "Decoder Total (bce_gen + bce_rec + gamma*recon)",
            "bce_gen": "BCE Gen",
            "bce_rec_dec": "BCE Rec (Decoder)",
            "dis_total": "Discriminator Total (bce_real + bce_fake + bce_rec)",
            "bce_real": "BCE Real",
            "bce_fake": "BCE Fake",
            "bce_rec_dis": "BCE Rec (Discriminator)",
        }

        keys = [k for k in self.loss_history.keys() if k not in ("step", "epoch")]
        ncols = 2
        nrows = ceil(len(keys) / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 3.2 * nrows), squeeze=False)
        steps = self.loss_history["step"]

        for idx, key in enumerate(keys):
            r = idx // ncols
            c = idx % ncols
            ax = axes[r][c]
            ax.plot(steps, self.loss_history[key], linewidth=1.6)
            ax.set_title(labels.get(key, key))
            ax.set_xlabel("Training step")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)
            ax.ticklabel_format(style="plain", axis="x")

        # Hide any empty subplots
        for idx in range(len(keys), nrows * ncols):
            r = idx // ncols
            c = idx % ncols
            axes[r][c].axis("off")

        model_name = "VAE-GAN"
        dataset = getattr(self.opt, "dataset", "unknown")
        fig.suptitle(f"{model_name} Loss Curves | dataset={dataset}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        out_path = os.path.join(self.opt.outf, "loss_curves.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        self.logger.info("Saved loss curves to %s", out_path)

    def train(self, dataloader):
        opt = self.opt
        train_dec = True
        train_dis = True
        torch.autograd.set_detect_anomaly(True)
        collapse_patience = 10
        self.patience = collapse_patience
        collapse = False
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
                self._set_requires_grad(self.netG, False)
                self._set_requires_grad(self.netD, False)
                x_enc = self.encoder(self.input_x)
                mu = x_enc[0]
                logvar = x_enc[1]
                if not torch.isfinite(mu).all() or not torch.isfinite(logvar).all():
                    self.logger.warning('Skipping batch due to non-finite encoder outputs')
                    continue
                # logvar = torch.clamp(logvar, min=-10.0, max=10.0)
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
                KLD_element = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
                KLD_loss = torch.mean(KLD_element) # torch.sum(KLD_element) previously 
                Disllike_loss = self._recon_loss(rec, rec_feats)
                VAEerr = opt.kld_wt * KLD_loss + Disllike_loss
                VAEerr.backward()
                self.optimizerEnc.step()
                self._set_requires_grad(self.netG, True)
                self._set_requires_grad(self.netD, True)
                ############################
                # (5) Calculate loss for Generator / Decoder
                ###########################
                self.netG.zero_grad()
                self.netD.zero_grad()
                # self._set_requires_grad(self.netG, True)
                sampled_detached = sampled.detach()
                rec = self.netG(sampled_detached)
                self._set_requires_grad(self.netD, False)
                rec_output, rec_feats = self.netD(rec)
                # self.logger.info(f"Reconstruction outputs | {rec_feats is None}")

                with torch.no_grad():
                    self.noise.resize_(batch_size, opt.nz, 1, 1)
                    self.noise.normal_(0, 1)
                gen = self.netG(self.noise)
                gen_output, _ = self.netD(gen)

                label_gen = torch.full_like(gen_output, float(self.real_label))
                bce_gen = self.criterion(gen_output, label_gen)
                label_rec = torch.full_like(rec_output, float(self.real_label))
                bce_rec_dec = self.criterion(rec_output, label_rec)
                Disllike_loss = self._recon_loss(rec, rec_feats)
                Decerr = bce_gen + bce_rec_dec + opt.gamma * Disllike_loss # bce_rec_dec
                # Decerr.backward()
                # self.optimizerG.step()
                #################################
                # (6) Calculate Loss for Discriminator
                ###########################
                self._set_requires_grad(self.netD, True)
                self.netD.zero_grad()
                output, _ = self.netD(self.input_x)
                label_real = torch.full_like(output, float(self.real_label))
                bce_real = self.criterion(output, label_real)
                D_x = output.data.mean()
                ############################
                output_fake, _ = self.netD(gen.detach())
                label_fake = torch.full_like(output_fake, float(self.fake_label))
                bce_fake = self.criterion(output_fake, label_fake)
                D_G_z1 = output_fake.data.mean()
                ############################
                output_rec, _ = self.netD(rec.detach())
                label_rec = torch.full_like(output_rec, float(self.fake_label))
                bce_rec_dis = self.criterion(output_rec, label_rec)
                D_G_z_rec = output_rec.data.mean()
                ############################
                Diserr = bce_real + bce_fake + bce_rec_dis
                # Diserr.backward()
                # self.optimizerD.step()
                #################################
                # selectively disable the decoder or the discriminator if they are unbalanced
                #################################
                if torch.mean(bce_real).item() < (opt.equillibrium - opt.margin) \
                    or torch.mean(bce_fake).item() < (opt.equillibrium - opt.margin):
                    train_dis = False
                if torch.mean(bce_real).item() > (opt.equillibrium + opt.margin) \
                    or torch.mean(bce_fake).item() > (opt.equillibrium + opt.margin):
                    train_dec = False
                if train_dec is False and train_dis is False:
                    train_dis = True
                    train_dec = True
                #############################
                if train_dec:
                    Decerr.backward()
                    self.optimizerG.step()
                    self.netD.zero_grad()
                if train_dis:
                    Diserr.backward()
                    self.optimizerD.step()
                #############################
                global_step = epoch * len(dataloader) + i
                self._record_losses(
                    epoch=epoch + 1,
                    step=global_step,
                    vae_total=VAEerr.item(),
                    kld=KLD_loss.item(),
                    recon=Disllike_loss.item(),
                    dec_total=Decerr.item(),
                    bce_gen=bce_gen.item(),
                    bce_rec_dec=bce_rec_dec.item(),
                    dis_total=Diserr.item(),
                    bce_real=bce_real.item(),
                    bce_fake=bce_fake.item(),
                    bce_rec_dis=bce_rec_dis.item(),
                )

                opt.margin *= opt.decay_margin
                opt.equillibrium *= opt.decay_equillibrium
                # margin non puo essere piu alto di equillibrium
                if opt.margin > opt.equillibrium:
                    opt.equillibrium = opt.margin
                opt.lambda_mse *= opt.decay_mse
                if opt.lambda_mse > 1:
                    opt.lambda_mse=1
                #############################
                if i % 500 == 0:
                    # save generated batch to file
                    self.logger.info(f"Encoder outputs | mu: {mu.mean().item():.4f} ± {mu.std().item():.4f}, logvar: {logvar.mean().item():.4f} ± {logvar.std().item():.4f}")
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

                if i == len(dataloader) - 1:
                    mu_mean = mu.mean().abs().item()
                    mu_var = mu.var(unbiased=False).item()
                    logvar_mean = logvar.mean().abs().item()
                    logvar_var = logvar.var(unbiased=False).item()
                    if (
                        mu_mean < 1e-4
                        and mu_var < 1e-4
                        and logvar_mean < 1e-4
                        and logvar_var < 1e-4
                    ):
                        self.patience -= 1
                        self.logger.info(
                            'Encoder collapse suspected (mean/var < 1e-4). Patience: %d',
                            self.patience,
                        )
                        if self.patience < 0:
                            self.logger.info(
                                'Stopping due to encoder collapse at epoch %d',
                                epoch + 1,
                            )
                            collapse = True
                            break
                    else:
                        self.patience = collapse_patience

            if collapse:
                break

            if (epoch + 1) % opt.saveInt == 0 and epoch != 0:
                torch.save(self.netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch + 1))
                torch.save(self.netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch + 1))
                torch.save(self.encoder.state_dict(), '%s/encoder_epoch_%d.pth' % (opt.outf, epoch + 1))

            self._step_schedulers()

        self._plot_losses()
        self.last_eval_metrics = self.evaluate(dataloader)
        return self.last_eval_metrics
