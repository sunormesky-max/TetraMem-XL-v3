"""
Multimodal bridge for TetraMesh.

Connects PixHomology (image, audio, video) to the tetrahedral mesh
by mapping media content to 3D geometric anchors that can be stored
as tetrahedra.

Per the v2.0 spec:
  - Image/Video/Audio: PixHomology extracts 0-dimensional topological
    features → mapped to 3D geometric anchor points
  - These anchors become tetrahedra in the mesh
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .multimodal import PixHomology
from .tetra_mesh import TetraMesh


class MultimodalBridge:
    def __init__(self, mesh: TetraMesh, pix_resolution: int = 32):
        self.mesh = mesh
        self.pix = PixHomology(resolution=pix_resolution)

    def store_image(
        self,
        image: np.ndarray,
        caption: str = "",
        labels: Optional[List[str]] = None,
        weight: float = 1.0,
    ) -> str:
        anchor = self.pix.image_to_geometry(image)
        content = caption or f"[image {image.shape}]"
        return self.mesh.store(
            content=content,
            seed_point=anchor,
            labels=(labels or []) + ["__image__"],
            metadata={"type": "image", "shape": list(image.shape)},
            weight=weight,
        )

    def store_image_tetrahedron(
        self,
        image: np.ndarray,
        caption: str = "",
        labels: Optional[List[str]] = None,
        weight: float = 1.0,
    ) -> str:
        vertices = self.pix.image_to_tetrahedron(image)
        anchor = np.mean(vertices, axis=0)
        content = caption or f"[image_tetra {image.shape}]"
        return self.mesh.store(
            content=content,
            seed_point=anchor,
            labels=(labels or []) + ["__image_tetra__"],
            metadata={
                "type": "image_tetrahedron",
                "shape": list(image.shape),
                "vertices": vertices.tolist(),
            },
            weight=weight,
        )

    def store_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 22050,
        caption: str = "",
        labels: Optional[List[str]] = None,
        weight: float = 1.0,
    ) -> str:
        anchor = self.pix.audio_to_geometry(audio_data, sample_rate)
        duration = len(audio_data) / sample_rate
        content = caption or f"[audio {duration:.1f}s]"
        return self.mesh.store(
            content=content,
            seed_point=anchor,
            labels=(labels or []) + ["__audio__"],
            metadata={"type": "audio", "duration": duration, "sample_rate": sample_rate},
            weight=weight,
        )

    def store_video(
        self,
        frames: list,
        fps: float = 30.0,
        caption: str = "",
        labels: Optional[List[str]] = None,
        weight: float = 1.0,
    ) -> str:
        anchor = self.pix.video_to_geometry(frames, fps)
        duration = len(frames) / fps
        content = caption or f"[video {duration:.1f}s {len(frames)}frames]"
        return self.mesh.store(
            content=content,
            seed_point=anchor,
            labels=(labels or []) + ["__video__"],
            metadata={"type": "video", "duration": duration, "fps": fps},
            weight=weight,
        )

    def query_by_modality(
        self,
        modality: str,
        query_point: Optional[np.ndarray] = None,
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        label_map = {
            "image": "__image__",
            "image_tetra": "__image_tetra__",
            "audio": "__audio__",
            "video": "__video__",
        }
        label = label_map.get(modality)
        if label is None:
            return []

        with self.mesh._lock:
            candidates = []
            for tid, tetra in self.mesh.tetrahedra.items():
                if label in tetra.labels:
                    if query_point is not None:
                        dist = float(np.linalg.norm(query_point - tetra.centroid))
                        candidates.append((tid, dist))
                    else:
                        candidates.append((tid, -float(tetra.weight)))
            candidates.sort(key=lambda x: x[1])
            return [(tid, -score) for tid, score in candidates[:k]]
