import os


_base_ = ["./infer_dataset_424full.py"]

# Full validation-set inference used for the paper evaluation. The generated
# RGB/depth/flow latents are retained so decoder variants can be evaluated
# without rerunning diffusion.
validation_index = "all"
dataset_start_on_firstframe = False
num_frames = 17
save_latents = True
save_latents_only = True
save_mode = "all-in-one"

model = dict(
    from_pretrained=os.environ.get(
        "DRIVE2GAUSS_CHECKPOINT",
        "checkpoints/drive2gauss/ema.pt",
    ),
)

outputs = os.environ.get("DRIVE2GAUSS_INFERENCE_ROOT", "outputs/inference")
