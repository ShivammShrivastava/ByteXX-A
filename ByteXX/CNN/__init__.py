"""
CNN Module — SatQuery AI
========================
Provides fast CNN-based analysis tools:
  - YOLOv8 object detection + square bounding box overlay
  - ResNet/U-Net pixel-level land cover segmentation

Usage:
    from CNN.cnn_pipeline import cnn_pipeline
    result = await cnn_pipeline.run(image, tasks=["detect", "segment"])
"""

from CNN.cnn_pipeline import CNNPipeline, cnn_pipeline

__all__ = ["CNNPipeline", "cnn_pipeline"]
