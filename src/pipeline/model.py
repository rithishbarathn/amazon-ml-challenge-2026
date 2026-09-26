"""Encoder network: two sparse EmbeddingBag layers (name, address)."""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.common import WORK_DIR
from pipeline.textvec import N_FEATURES

# Per-field dimension (stored vector is 2 * DIM) and checkpoint path; both can be
# overridden for encoder experiments (ER_ENC_DIM / ER_ENC_PATH).
DIM = int(os.environ.get("ER_ENC_DIM", 96))
MODEL_PATH = Path(os.environ.get("ER_ENC_PATH", WORK_DIR / "encoder.pt"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class Encoder(nn.Module):

    def __init__(self, dim=DIM):
        super().__init__()
        self.name_bag = nn.EmbeddingBag(N_FEATURES, dim, mode="sum", sparse=True)
        self.addr_bag = nn.EmbeddingBag(N_FEATURES, dim, mode="sum", sparse=True)
        nn.init.normal_(self.name_bag.weight, std=0.1)
        nn.init.normal_(self.addr_bag.weight, std=0.1)
        # Relative weight of the address in the combined vector, and the
        # InfoNCE temperature (both learned, stored in log space).
        self.log_addr_w = nn.Parameter(torch.zeros(()))
        self.log_tau = nn.Parameter(torch.tensor(np.log(0.05), dtype=torch.float32))

    @staticmethod
    def _bag(layer, csr):
        indices = torch.from_numpy(csr.indices.astype(np.int64)).to(DEVICE)
        offsets = torch.from_numpy(csr.indptr[:-1].astype(np.int64)).to(DEVICE)
        weights = torch.from_numpy(csr.data).to(DEVICE)
        return layer(indices, offsets, per_sample_weights=weights)

    def fields(self, name_csr, addr_csr):
        name = F.normalize(self._bag(self.name_bag, name_csr), dim=1, eps=1e-6)
        addr = F.normalize(self._bag(self.addr_bag, addr_csr), dim=1, eps=1e-6)
        return name, addr

    def combine(self, name, addr):
        return combine(name, addr, self.log_addr_w.exp())


def combine(name, addr, addr_w):
    """Combined unit vector used for nearest-neighbour search."""
    return F.normalize(torch.cat([name, addr * addr_w], dim=1), dim=1)


def load_encoder(path=None):
    state = torch.load(path or MODEL_PATH, map_location=DEVICE)
    model = Encoder(dim=state["name_bag.weight"].shape[1]).to(DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


