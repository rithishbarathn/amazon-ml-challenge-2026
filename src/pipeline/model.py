"""Encoder network: two sparse EmbeddingBag layers (name, address)."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.common import WORK_DIR
from pipeline.textvec import N_FEATURES

DIM = 96                      # per field -> stored vector is 2 * DIM
MODEL_PATH = WORK_DIR / "encoder.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.name_bag = nn.EmbeddingBag(N_FEATURES, DIM, mode="sum", sparse=True)
        self.addr_bag = nn.EmbeddingBag(N_FEATURES, DIM, mode="sum", sparse=True)
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


def load_encoder():
    model = Encoder().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


