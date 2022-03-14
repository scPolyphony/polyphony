import torch

from scvi.module import VAE
from scvi.module.base import LossRecorder


class ActiveVAE(VAE):
    def __init__(self, *args, **kwargs):
        super(ActiveVAE, self).__init__(*args, **kwargs)

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
        cell_update = tensors['cell_update'].squeeze().type(torch.bool)
        desired_rep = tensors['desired_rep'][cell_update]
        rep = inference_outputs['z'][cell_update]

        anchor_loss = torch.linalg.norm(desired_rep - rep, dim=1).sum()

        old_loss_record = VAE.loss(
            self,
            tensors,
            inference_outputs,
            generative_outputs,
            kl_weight
        )

        loss = old_loss_record.loss
        reconst_loss = old_loss_record.reconstruction_loss
        kl_local = old_loss_record.kl_local
        kl_global = old_loss_record.kl_global

        loss += anchor_loss

        return LossRecorder(loss, reconst_loss, kl_local, kl_global, anchor_loss=anchor_loss)
