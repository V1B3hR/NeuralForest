"""
tree_graveyard.py — backward-compatibility shim.

The canonical implementation has been renamed to ``evolution/humus_nursery.py`` as
part of Phase 1 (Soil Purification & Rebirth) of the Tropical Forest Map.
All public symbols are re-exported here so that existing import sites continue to work
without modification.
"""

from .humus_nursery import (  # noqa: F401
    TreeRecord,
    NurseryStats as GraveyardStats,
    HumusNursery as TreeGraveyard,
)

__all__ = ["TreeRecord", "GraveyardStats", "TreeGraveyard"]
