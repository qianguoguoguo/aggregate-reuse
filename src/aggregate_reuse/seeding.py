import hashlib
import random
from typing import Optional

try:
    import numpy as np
except Exception:
    np = None


def derive_stage_seed(master_seed: int, stage_name: str) -> int:
    """
    Deterministically derive a 32-bit stage-specific seed from
    (master_seed, stage_name). This prevents accidental coupling between
    preprocessing, calibration, intervention, identity assignment, etc.
    """
    payload = f"{int(master_seed)}::{stage_name}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def seed_everything(master_seed: int, stage_name: str) -> int:
    stage_seed = derive_stage_seed(master_seed, stage_name)
    random.seed(stage_seed)
    if np is not None:
        np.random.seed(stage_seed)
    return stage_seed
