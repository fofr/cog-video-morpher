import os
import shutil
import random
import json
import shutil
from PIL import Image
from typing import List
from cog import BasePredictor, Input, Path
from helpers.comfyui import ComfyUI

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"

with open("video_workflow_api.json", "r") as file:
    WORKFLOW_JSON = file.read()


class Predictor(BasePredictor):
    aspect_ratio_map = {
        "16:9": {
            "small": {"width": 512, "height": 288},
            "full": {"width": 1920, "height": 1080},
        },
        "4:3": {
            "small": {"width": 440, "height": 328},
            "full": {"width": 1664, "height": 1248},
        },
        "3:2": {
            "small": {"width": 464, "height": 312},
            "full": {"width": 1768, "height": 1176},
        },
        "1:1": {
            "small": {"width": 384, "height": 384},
            "full": {"width": 1440, "height": 1440},
        },
        "2:3": {
            "small": {"width": 312, "height": 464},
            "full": {"width": 1176, "height": 1768},
        },
        "3:4": {
            "small": {"width": 328, "height": 440},
            "full": {"width": 1248, "height": 1664},
        },
        "9:16": {
            "small": {"width": 288, "height": 512},
            "full": {"width": 1080, "height": 1920},
        },
    }

    style_checkpoint_map = {
        "realistic": "juggernaut_reborn.safetensors",
        "illustrated": "dreamshaper_8.safetensors",
        "anime": "toonyou_beta6.safetensors",
        "3D": "rcnzCartoon3d_v20",
        "any": "Deliberate_v2.safetensors",
    }

    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

    def cleanup(self):
        self.comfyUI.clear_queue()
        for directory in [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)

    def handle_input_file(self, input_file: Path, filename: str = "image.png"):
        image = Image.open(input_file)
        image.save(os.path.join(INPUT_DIR, filename))

    def log_and_collect_files(self, directory, prefix=""):
        files = []
        for f in os.listdir(directory):
            if f == "__MACOSX" or not f.endswith(".mp4"):
                continue
            path = os.path.join(directory, f)
            if os.path.isfile(path):
                print(f"{prefix}{f}")
                files.append(Path(path))
            elif os.path.isdir(path):
                print(f"{prefix}{f}/")
                files.extend(self.log_and_collect_files(path, prefix=f"{prefix}{f}/"))
        return files

    def update_workflow(self, workflow, **kwargs):
        workflow["565"]["inputs"]["text"] = kwargs["prompt"]
        workflow["566"]["inputs"]["text"] = f"nsfw, nude, {kwargs['negative_prompt']}"

        # best results if seed for second sampler is different
        # but we can still make it deterministic
        workflow["80"]["seed"] = kwargs["seed"]
        workflow["198"]["seed"] = kwargs["seed"] + 10

        sizes = self.aspect_ratio_map[kwargs["aspect_ratio"]]
        checkpoint_filename = self.style_checkpoint_map[kwargs["checkpoint"]]

        workflow["564"]["inputs"]["ckpt_name"] = checkpoint_filename

        initial_empty_latent_image = workflow["134"]["inputs"]
        initial_empty_latent_image["width"] = sizes["small"]["width"]
        initial_empty_latent_image["height"] = sizes["small"]["height"]

        upscaled_dimensions = workflow["279"]["inputs"]
        upscaled_dimensions["width"] = sizes["full"]["width"]
        upscaled_dimensions["height"] = sizes["full"]["height"]

        style_ip_adapter = workflow["751"]["inputs"]
        if kwargs["has_style_image"]:
            style_ip_adapter["weight"] = kwargs["style_strength"]
            style_ip_adapter["end_at"] = 1
        else:
            style_ip_adapter["weight"] = 0
            style_ip_adapter["end_at"] = 0

        mode = kwargs["mode"]

        if mode == "small":
            # disable latent upscaling
            del workflow["198"]
            del workflow["201"]["samples"]
            workflow["80"]["positive"] = ["125", 0]
            workflow["80"]["negative"] = ["125", 1]
        elif mode == "medium":
            # disable upscaling
            del workflow["271"]
            del workflow["279"]["inputs"]["image"]
        elif mode == "upscaled":
            del workflow["770"]
            # node 219 has "images"
            del workflow["219"]["images"]
        elif mode == "upscaled-and-interpolated":
            # default
            pass

    def predict(
        self,
        prompt: str = Input(
            description="The prompt has a small effect, but most of the video is driven by the subject images",
            default="",
        ),
        negative_prompt: str = Input(
            description="What you do not want to see in the video",
            default="",
        ),
        aspect_ratio: str = Input(
            description="The aspect ratio of the video",
            default="2:3",
            choices=["16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16"],
        ),
        mode: str = Input(
            description="Determines if you produce a quick experimental video or an upscaled interpolated one.",
            default="medium",
            choices=["small", "medium", "upscaled", "upscaled-and-interpolated"],
        ),
        subject_image_1: Path = Input(
            description="The first subject of the video",
        ),
        subject_image_2: Path = Input(
            description="The second subject of the video",
        ),
        subject_image_3: Path = Input(
            description="The third subject of the video",
        ),
        subject_image_4: Path = Input(
            description="The fourth subject of the video",
        ),
        style_image: Path = Input(
            description="Apply the style from this image to the whole video",
            default=None,
        ),
        style_strength: float = Input(
            description="How strong the style is applied",
            default=1.0,
            ge=0.0,
            le=2.0,
        ),
        checkpoint: str = Input(
            description="The checkpoint to use for the model",
            default="realistic",
            choices=[
                "realistic",
                "illustrated",
                "anime",
                "3D",
                "any",
            ],
        ),
        seed: int = Input(
            description="Set a seed for reproducibility. Random by default.",
            default=None,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.cleanup()
        shutil.copy("circles.mp4", f"{INPUT_DIR}/circles.mp4")

        for i, image in enumerate(
            [subject_image_1, subject_image_2, subject_image_3, subject_image_4]
        ):
            self.handle_input_file(image, f"{i + 1}.png")

        if style_image:
            self.handle_input_file(style_image, "style.png")
        else:
            self.handle_input_file(subject_image_1, "style.png")

        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            print(f"Random seed set to: {seed}")

        workflow = json.loads(WORKFLOW_JSON)
        self.update_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            aspect_ratio=aspect_ratio,
            mode=mode,
            seed=seed,
            has_style_image=style_image is not None,
            style_strength=style_strength,
            checkpoint=checkpoint,
        )

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)
        files = self.log_and_collect_files(OUTPUT_DIR)
        return files
