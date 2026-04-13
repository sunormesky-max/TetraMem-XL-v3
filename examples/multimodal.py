"""Multimodal demo: image, audio, and video to geometry via PixHomology."""

import numpy as np

from tetrahedron_memory import PixHomology


def main():
    pix = PixHomology(resolution=16)  # Low res for fast demo

    # --- Image to geometry ---
    image = np.random.rand(16, 16, 3)
    geometry = pix.image_to_geometry(image)
    print(
        f"Image -> geometry: shape={geometry.shape}, range=[{geometry.min():.3f}, {geometry.max():.3f}]"
    )

    # --- Audio to geometry ---
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    audio_geometry = pix.audio_to_geometry(audio, sample_rate=sample_rate)
    print(f"Audio -> geometry: shape={audio_geometry.shape}")

    # --- Video to geometry ---
    frames = [np.random.rand(16, 16, 3) for _ in range(8)]
    video_geometry = pix.video_to_geometry(frames, fps=24.0)
    print(f"Video -> geometry: shape={video_geometry.shape} (from {len(frames)} frames)")

    # --- Store multimodal in GeoMemoryBody ---
    from tetrahedron_memory import GeoMemoryBody

    memory = GeoMemoryBody(dimension=3, precision="fast")
    memory.store(
        content="image memory",
        labels=["multimodal", "image"],
        weight=1.0,
        geometry_override=geometry,
    )
    memory.store(
        content="audio memory",
        labels=["multimodal", "audio"],
        weight=1.0,
        geometry_override=audio_geometry,
    )
    memory.store(
        content="video memory",
        labels=["multimodal", "video"],
        weight=1.0,
        geometry_override=video_geometry,
    )
    print("\nStored 3 multimodal memories")

    results = memory.query_by_label("multimodal")
    print(f"Label query 'multimodal': {len(results)} results")
    for node in results:
        print(f"  {node.content} | labels={node.labels} | weight={node.weight:.2f}")


if __name__ == "__main__":
    main()
