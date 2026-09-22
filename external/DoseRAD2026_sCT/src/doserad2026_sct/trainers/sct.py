"""MR -> sCT generator training: SwinUNETR generator + PatchGAN discriminator.

The objective is a body-masked MSE reconstruction loss plus an LSGAN adversarial
term -- exactly what trained the two generators the submission serves. The
discriminator is unconditional: it sees only the CT/sCT, never the MR.

The two validation *diagnostics* are not losses; they are how a run is read.
Whole-volume validation L1 is a poor guide on its own -- enclosed air is ~1.7 %
of a thorax volume but roughly half the beam-level dose error, and a model can
improve per-voxel HU while getting the accumulated path, and so the depth-dose
curve, worse. So each epoch also records signed HU bias inside enclosed air and
transverse water-equivalent path error.
"""

import os

from monai.inferers import sliding_window_inference
import numpy as np
import SimpleITK as sitk
import torch
import tqdm

from doserad2026_sct.utils import masks, normalisation, preprocess


class sCTTrainer:
    def __init__(
        self,
        config: dict[str, str | float | int],
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        validation_dataloader: torch.utils.data.DataLoader,
        device: torch.device | str | int,
        out_path: str,
        resume_training: bool,
    ) -> None:
        self.config = config
        self.generator = generator
        self.discriminator = discriminator
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.device = device
        self.out_path = out_path
        self.resume_training = resume_training

        # Dumping every validation volume every 10 epochs was hardcoded; at
        # ~3 patients x 100 dumps it is a lot of disk for images nobody reads.
        # 0 disables. Set e.g. 100 to keep them aligned with the snapshots.
        self.validation_volume_freq = int(self.config.get("validation_volume_freq", 0))
        # Periodic generator-only checkpoints, so a run can be re-scored at
        # epochs other than the single best-val-L1 one. Generator state is
        # 285 MB against 870 MB for the full resumable snapshot, and nothing
        # downstream (stage_sct_generators.py, doserad.sct.SCTGenerator) needs
        # the optimizer states.
        self.snapshot_freq = int(self.config.get("snapshot_freq", 100))

        self._parse_generator_criterion()
        self._parse_generator_optimizer()
        self._parse_discriminator_criterion()
        self._parse_discriminator_optimizer()
        self._parse_learning_rate_scheduler()
        self._parse_validation_metric()

        if self.resume_training:
            self._load_snapshot()
            self._load_losses()

    def _parse_generator_criterion(self) -> None:
        match self.config["generator_image_criterion"].lower():
            case "l1":
                self.generator_image_criterion = torch.nn.L1Loss()
            case "mse":
                self.generator_image_criterion = torch.nn.MSELoss()
            case _:
                raise NotImplementedError(
                    f"{self.config['generator_image_criterion']} not implemented"
                )

    def _parse_discriminator_criterion(self) -> None:
        match self.config["discriminator_criterion"].lower():
            case "mse":
                # LSGAN: least squares on the discriminator output. This *is*
                # the adversarial objective, not an incidental choice.
                self.discriminator_criterion = torch.nn.MSELoss()
            case _:
                raise NotImplementedError(
                    f"{self.config['discriminator_criterion']} not implemented"
                )

    def _parse_validation_metric(self) -> None:
        match self.config["validation_metric"].lower():
            case "l1":
                self.validation_metric = torch.nn.L1Loss()
            case "mse":
                self.validation_metric = torch.nn.MSELoss()
            case _:
                raise NotImplementedError(
                    f"{self.config['validation_metric']} not implemented"
                )

    def _parse_generator_optimizer(self) -> None:
        match self.config["generator_optimizer"].lower():
            case "adamw":
                self.generator_optimizer = torch.optim.AdamW(
                    self.generator.parameters(),
                    lr=self.config["generator_learning_rate"],
                    weight_decay=1e-4,
                )
            case "adam":
                self.generator_optimizer = torch.optim.Adam(
                    self.generator.parameters(),
                    lr=self.config["generator_learning_rate"],
                )
            case _:
                raise NotImplementedError(
                    f"{self.config['generator_optimizer']} not implemented"
                )

    def _parse_discriminator_optimizer(self) -> None:
        match self.config["discriminator_optimizer"].lower():
            case "adamw":
                self.discriminator_optimizer = torch.optim.AdamW(
                    self.discriminator.parameters(),
                    lr=self.config["discriminator_learning_rate"],
                    weight_decay=1e-4,
                )
            case "adam":
                self.discriminator_optimizer = torch.optim.Adam(
                    self.discriminator.parameters(),
                    lr=self.config["discriminator_learning_rate"],
                )
            case _:
                raise NotImplementedError(
                    f"{self.config['discriminator_optimizer']} not implemented"
                )

    def _parse_learning_rate_scheduler(self) -> None:
        match self.config["learning_rate_scheduler"].lower():
            case "reducelronplateau":
                self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.generator_optimizer,
                    patience=10,
                    factor=0.5,
                    min_lr=self.config["generator_learning_rate"] / 1e3,
                )
            case "none":
                self.lr_scheduler = None
            case _:
                raise NotImplementedError(
                    f"{self.config['learning_rate_scheduler']} not implemented"
                )

    def _save_snapshot(self, epoch_idx: int) -> None:
        snapshot = {}
        snapshot["GENERATOR_MODEL_STATE"] = self.generator.state_dict()
        snapshot["DISCRIMINATOR_MODEL_STATE"] = self.discriminator.state_dict()
        snapshot["GENERATOR_OPTIMIZER_STATE"] = self.generator_optimizer.state_dict()
        snapshot["DISCRIMINATOR_OPTIMIZER_STATE"] = (
            self.discriminator_optimizer.state_dict()
        )
        if self.lr_scheduler is not None:
            snapshot["LEARNING_RATE_SCHEDULER_STATE"] = self.lr_scheduler.state_dict()
        snapshot["EPOCH"] = epoch_idx
        torch.save(snapshot, os.path.join(self.out_path, "snapshot.pt"))

    def _save_generator_only(self, filename: str, epoch_idx: int) -> None:
        """Generator weights alone, for evaluation rather than resuming.

        Written in the same shape the full snapshot uses, so
        ``submission/stage_sct_generators.py`` and ``doserad.sct.SCTGenerator``
        both load it unchanged.
        """
        torch.save(
            {
                "GENERATOR_MODEL_STATE": self.generator.state_dict(),
                "EPOCH": epoch_idx,
            },
            os.path.join(self.out_path, filename),
        )

    def _load_snapshot(self) -> None:
        snapshot = torch.load(
            os.path.join(self.out_path, "snapshot.pt"), weights_only=False
        )
        self.generator.load_state_dict(snapshot["GENERATOR_MODEL_STATE"])
        self.discriminator.load_state_dict(snapshot["DISCRIMINATOR_MODEL_STATE"])
        self.generator_optimizer.load_state_dict(snapshot["GENERATOR_OPTIMIZER_STATE"])
        self.discriminator_optimizer.load_state_dict(
            snapshot["DISCRIMINATOR_OPTIMIZER_STATE"]
        )
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(snapshot["LEARNING_RATE_SCHEDULER_STATE"])
        self.start_epoch = snapshot["EPOCH"] + 1

    def _load_losses(self) -> None:
        # Histories are truncated to start_epoch: snapshot.pt is written only
        # when validation improves, so any epoch after it has no weights to
        # match and must be re-run.
        def _required(name: str) -> list[float]:
            # No fallback on purpose. These are written every epoch, so one
            # missing means the run directory is not the resumable state it
            # looks like -- better to fail here than to train on a silently
            # truncated history.
            return list(np.load(os.path.join(self.out_path, name))[: self.start_epoch])

        def _optional(name: str) -> list[float]:
            # Absent when resuming a run started before this diagnostic existed.
            path = os.path.join(self.out_path, name)
            if not os.path.isfile(path):
                return [float("nan")] * self.start_epoch
            return list(np.load(path)[: self.start_epoch])

        self.generator_image_train_losses = _required(
            "generator_image_train_losses.npy"
        )
        self.discriminator_train_losses = _required("discriminator_train_losses.npy")
        self.validation_metrics = _required("validation_metrics.npy")
        self.validation_low_density_bias = _optional(
            "validation_low_density_bias.npy"
        )
        self.validation_wep_error = _optional("validation_wep_error.npy")

    def _train_epoch(self, epoch_idx: int) -> None:
        self.generator.train()
        self.discriminator.train()
        generator_image_losses_epoch = []
        discriminator_losses_epoch = []
        for data, _ in tqdm.tqdm(self.train_dataloader, desc=f"EPOCH: {epoch_idx}"):
            mr = torch.cat(
                [data[p]["image"] for p in range(self.config["batch_size"])], dim=0
            ).to(self.device)
            ct = torch.cat(
                [data[p]["label"] for p in range(self.config["batch_size"])], dim=0
            ).to(self.device)
            mask = torch.cat(
                [data[p]["mask"] for p in range(self.config["batch_size"])], dim=0
            ).to(self.device)

            sct = self.generator(mr)

            # train discriminator
            real_prediction = self.discriminator(ct)
            fake_prediction = self.discriminator(sct.detach())
            real_loss = self.discriminator_criterion(
                real_prediction, torch.ones_like(real_prediction)
            )
            fake_loss = self.discriminator_criterion(
                fake_prediction, torch.zeros_like(fake_prediction)
            )
            discriminator_loss = (real_loss + fake_loss) * 0.5
            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizer.step()
            discriminator_losses_epoch.append(discriminator_loss.item())

            # train generator
            generator_image_loss = self.generator_image_criterion(
                sct * mask + (1 - mask) * -1, ct * mask + (1 - mask) * -1
            )
            real_or_fake_prediction = self.discriminator(sct)
            real_or_fake_loss = self.discriminator_criterion(
                real_or_fake_prediction, torch.ones_like(real_or_fake_prediction)
            )
            generator_loss = (
                self.config["lambda_generator_discriminator"] * generator_image_loss
                + real_or_fake_loss
            )

            self.generator_optimizer.zero_grad()
            generator_loss.backward()
            self.generator_optimizer.step()

            generator_image_losses_epoch.append(generator_image_loss.item())

        self.generator_image_train_losses.append(np.mean(generator_image_losses_epoch))
        self.discriminator_train_losses.append(np.mean(discriminator_losses_epoch))

    @torch.inference_mode()
    def _validation_epoch(self, epoch_idx: int) -> None:
        with torch.autocast("cuda", dtype=torch.float16):
            validation_metrics_epoch = []
            low_density_bias_epoch = []
            wep_epoch = []
            for mr, ct, mask, stats in tqdm.tqdm(
                self.validation_dataloader, desc=f"EPOCH: {epoch_idx}"
            ):
                self.generator.eval()
                mr = mr.unsqueeze(1)
                ct = ct.unsqueeze(1)
                sct = sliding_window_inference(
                    inputs=mr.to(self.device),
                    roi_size=self.config["patch_size"],
                    sw_batch_size=1,
                    predictor=self.generator,
                    overlap=0.5,
                )
                sct = normalisation.z_score_denormalise(
                    sct,
                    np.mean(stats["ct_population"]["mean"]),
                    np.mean(stats["ct_population"]["std"]),
                )
                sct = torch.where(
                    mask.to(self.device) == 0,
                    torch.as_tensor(
                        preprocess.CT_LOWEST_VALUE,
                        device=sct.device,
                        dtype=sct.dtype,
                    ),
                    sct,
                )
                sct = torch.clamp(
                    sct, preprocess.CT_LOWEST_VALUE, preprocess.CT_HIGHEST_VALUE
                )
                validation_metrics_epoch.append(
                    self.validation_metric(sct, ct.to(self.device)).item()
                )
                # Transverse water-equivalent path error, in mm of water: the
                # quantity IDD integrates. Reported alongside the HU metrics
                # because a model can improve per-voxel HU while getting the
                # accumulated path -- and therefore the depth-dose curve -- worse.
                sct_hu = sct.detach().float().cpu().numpy().squeeze()
                ct_hu_full = ct.cpu().numpy().squeeze()
                body_np = mask.cpu().numpy().squeeze() > 0
                rho_p = np.clip(1.0 + sct_hu / 1000.0, 0.0, None) * body_np
                rho_t = np.clip(1.0 + ct_hu_full / 1000.0, 0.0, None) * body_np
                spacing_mm = 2.0
                wep_err = [
                    float(
                        np.abs(rho_p.sum(axis=ax) - rho_t.sum(axis=ax)).mean()
                        * spacing_mm
                    )
                    for ax in (1, 2)
                ]
                wep_epoch.append(float(np.mean(wep_err)))
                # Signed HU bias inside enclosed air. Global L1 cannot see this
                # -- cavities are ~1.7 % of voxels -- but it is roughly half of
                # the beam-level dose error.
                ct_np = ct.cpu().numpy().squeeze()
                low_density_np = masks.get_enclosed_low_density_mask(ct_np)
                if low_density_np.any():
                    low_density_bias_epoch.append(
                        float((sct_hu[low_density_np] - ct_np[low_density_np]).mean())
                    )
                mean_validation_metric = np.mean(validation_metrics_epoch)
                if (
                    self.validation_volume_freq
                    and epoch_idx % self.validation_volume_freq == 0
                ):
                    sct_out = sct.cpu().numpy().squeeze().astype(np.float32)
                    sct_sitk = sitk.GetImageFromArray(sct_out)
                    os.makedirs(
                        os.path.join(self.out_path, f"epoch_{epoch_idx}"), exist_ok=True
                    )
                    sitk.WriteImage(
                        sct_sitk,
                        os.path.join(
                            self.out_path,
                            f"epoch_{epoch_idx}",
                            f"{stats['patient_id'][0]}.mha",
                        ),
                    )
        self.validation_metrics.append(mean_validation_metric)
        self.validation_low_density_bias.append(
            float(np.mean(low_density_bias_epoch))
            if low_density_bias_epoch
            else float("nan")
        )
        self.validation_wep_error.append(
            float(np.mean(wep_epoch)) if wep_epoch else float("nan")
        )
        tqdm.tqdm.write(
            f"  val L1 {mean_validation_metric:.2f} HU | "
            f"enclosed-air bias {self.validation_low_density_bias[-1]:+.1f} HU | "
            f"transverse WEP error {self.validation_wep_error[-1]:.2f} mm"
        )
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(mean_validation_metric)

    def train(self) -> None:
        if not self.resume_training:
            self.generator_image_train_losses = []
            self.discriminator_train_losses = []
            self.validation_metrics = []
            self.validation_low_density_bias = []
            self.validation_wep_error = []
            self.start_epoch = 0

        for epoch in range(self.start_epoch, self.config["maximum_epochs"]):
            self._train_epoch(epoch_idx=epoch)
            self._validation_epoch(epoch_idx=epoch)
            # snapshot.pt tracks the best validation L1 -- this is the selection
            # rule that produced both shipped checkpoints (thorax epoch 972,
            # abdomen epoch 914).
            if np.min(self.validation_metrics) == self.validation_metrics[-1]:
                print("validation loss decreased, saving snapshot")
                self._save_snapshot(epoch_idx=epoch)

            if self.snapshot_freq and epoch % self.snapshot_freq == 0:
                self._save_generator_only(f"snapshot_epoch_{epoch:04d}.pt", epoch)

            for name, series in (
                ("generator_image_train_losses.npy", self.generator_image_train_losses),
                ("discriminator_train_losses.npy", self.discriminator_train_losses),
                ("validation_metrics.npy", self.validation_metrics),
                ("validation_low_density_bias.npy", self.validation_low_density_bias),
                ("validation_wep_error.npy", self.validation_wep_error),
            ):
                np.save(os.path.join(self.out_path, name), series)
