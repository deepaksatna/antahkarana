"""
ChittaKit — the novel modules of the Antaḥkaraṇa-Net.

A thin research library (PyTorch-first) implementing the ~5 pieces that do not
exist off-the-shelf, on top of mature deps. See ../README.md and the design docs
in the parent `vedas/` folder.

    SamskaraMemory   — continual memory with importance growth + decay (chitta)
    GunaController   — one (sattva,rajas,tamas) signal modulating all learning
    AshramaSchedule  — lifelong plasticity, re-openable critical periods
    WitnessMonitor   — task-reward-invariant identity/consistency overseer (turīya)
    klesha           — affliction metrics (avidyā = epistemic uncertainty)
"""
from .samskara import SamskaraMemory
from .guna import GunaController, GunaState
from .meta_guna import MetaGunaController
from .ashrama import AshramaSchedule, AshramaState, effective_alpha, STAGES
from .tapas import TapasController
from .pramana import PramanaGate
from .witness import WitnessMonitor
from .antahkarana import Antahkarana
from . import klesha

__all__ = [
    "SamskaraMemory",
    "GunaController", "GunaState", "MetaGunaController",
    "AshramaSchedule", "AshramaState", "effective_alpha", "STAGES",
    "TapasController",
    "PramanaGate",
    "WitnessMonitor",
    "Antahkarana",
    "klesha",
]

__version__ = "0.1.0"
