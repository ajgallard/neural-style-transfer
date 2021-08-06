"""
Download Script for Coco Dataset for Train 2014
FiftyOne import method
"""

import fiftyone as fo
import fiftyone.zoo as foz

# Full Train split dataset has 82,783 images
dataset = foz.load_zoo_dataset(
    "coco-2014",
    split="train",
    max_samples=50000,
    shuffle=True,
)

session = fo.launch_app(dataset)