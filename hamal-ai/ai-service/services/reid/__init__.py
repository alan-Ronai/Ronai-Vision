"""ReID (Re-Identification) services for tracking persons and vehicles across cameras."""

from .base_reid import BaseReID
from .osnet_reid import OSNetReID
from .universal_reid import UniversalReID
from .reid_tracker import ReIDTracker
from .transreid_vehicle import TransReIDVehicle
from .reid_gallery import (
    ReIDGallery,
    GalleryEntry,
    get_reid_gallery,
    initialize_reid_gallery,
    reset_reid_gallery,
)

__all__ = [
    "BaseReID",
    "OSNetReID",
    "UniversalReID",
    "ReIDTracker",
    "TransReIDVehicle",
    "ReIDGallery",
    "GalleryEntry",
    "get_reid_gallery",
    "initialize_reid_gallery",
    "reset_reid_gallery",
]
