from models.component_segmentaion import grounding_segmentation
import os
import glob
import yaml


mask_path = "./masks/madsim"
categories = [
    '01Gorilla', '03Mallard', '05Whale', '07Owl', '09Swan', '11Pig', '13Pheonix', '15Parrot', '17Scorpion', '19Bear',
    '02Unicorn', '04Turtle', '06Bird', '08Sabertooth', '10Sheep', '12Zalika', '14Elephant', '16Cat', '18Obesobeso', '20Puppy',
]


# def read_config(config_path):
#     with open(config_path, "r") as f:
#         config = yaml.load(f, Loader=yaml.SafeLoader)
#     return config


# grounding_config:
#   box_threshold: 0.15
#   text_threshold: 0.15
#   text_prompt: "capsule . " # . grain . mixture
#   background_prompt: ""

config = {
    "box_threshold": 0.15,
    "text_threshold": 0.15,
    "text_prompt": "capsule . ",
    "background_prompt": "",
}

for category in categories:
    image_paths = sorted(glob.glob(f"./data/madsim/{category}/test/*/*.png"))
    this_config = config.copy()
    this_config["text_prompt"] = f"{category[2:].lower()} . "
    os.makedirs(f"{mask_path}/{category}", exist_ok=True)
    grounding_segmentation(
        image_paths, f"{mask_path}/{category}", this_config
    )

for category in categories:
    image_paths = sorted(glob.glob(f"./data/madsim/{category}/train/*/*.png"))
    this_config = config.copy()
    this_config["text_prompt"] = f"{category[2:].lower()} . "
    os.makedirs(f"{mask_path}/{category}", exist_ok=True)
    grounding_segmentation(
        image_paths, f"{mask_path}/{category}", this_config
    )

