import os
import re
import time
from dataclasses import dataclass
from glob import iglob
import argparse
import torch
from einops import rearrange
from PIL import ExifTags, Image

from flux.sampling import denoise_midpoint,denoise_fireflow, denoise_fireflow_M, denoise_rf_solver, denoise, denoise_m, get_schedule, prepare, unpack,get_schedule, prepare, unpack
from flux.util import (configs, embed_watermark, load_ae, load_clip,
                       load_flow_model, load_t5, save_velocity_distribution)
from transformers import pipeline
from PIL import Image
import numpy as np

import os

NSFW_THRESHOLD = 0.85


@dataclass
class SamplingOptions:
    source_prompt: str
    target_prompt: str
    # prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None


@torch.inference_mode()
def encode(init_image, torch_device, ae):
    init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
    init_image = init_image.unsqueeze(0)
    init_image = init_image.to(torch_device)
    init_image = ae.encode(init_image.to()).to(torch.bfloat16)
    return init_image


# @torch.inference_mode()
def main(
        args,
        seed: int | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_steps: int | None = None,
        loop: bool = False,
        offload: bool = False,
        add_sampling_metadata: bool = True,
):
    
    torch.set_grad_enabled(False)
    name = args.name
    source_prompt = args.source_prompt
    target_prompt = args.target_prompt
    guidance = args.guidance
    output_dir = args.output_dir
    num_steps = args.num_steps
    offload = args.offload
    prefix = args.output_prefix
    inject = args.inject
    start_layer_index = args.start_layer_index
    end_layer_index = args.end_layer_index
    seed = args.seed if args.seed > 0 else None

    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)

    if name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Got unknown model name: {name}, chose from {available}")

    torch_device = torch.device(device)
    if num_steps is None:
        num_steps = 4 if name == "flux-schnell" else 25

    # init all components
    t5 = load_t5(torch_device, max_length=256 if name == "flux-schnell" else 512)
    clip = load_clip(torch_device)
    model = load_flow_model(name, device="cpu" if offload else torch_device)
    ae = load_ae(name, device="cpu" if offload else torch_device)

    if offload:
        model.cpu()
        torch.cuda.empty_cache()
        ae.encoder.to(torch_device)

    init_image = None
    init_image_array = np.array(Image.open(args.source_img_dir).convert('RGB'))
    shape = init_image_array.shape

    new_h = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
    new_w = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16

    init_image = init_image_array[:new_h, :new_w, :]
    width, height = init_image.shape[0], init_image.shape[1]

    t0 = time.perf_counter()

    init_image = encode(init_image, torch_device, ae)

    rng = torch.Generator(device="cpu")
    opts = SamplingOptions(
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
    )

    if loop:
        opts = parse_prompt(opts)

    while opts is not None:
        if opts.seed is None:
            opts.seed = rng.seed()
        print(f"Generating with seed {opts.seed}:\n{opts.source_prompt}")

        opts.seed = None
        if offload:
            ae = ae.cpu()
            torch.cuda.empty_cache()
            t5, clip = t5.to(torch_device), clip.to(torch_device)

        info = {}
        info['feature_path'] = args.feature_path
        info['feature'] = {}
        info['inject_step'] = inject
        info['start_layer_index'] = start_layer_index
        info['end_layer_index'] = end_layer_index
        info['reuse_v'] = args.reuse_v
        info['editing_strategy'] = args.editing_strategy
        info['qkv_ratio'] = list(map(float, args.qkv_ratio.split(',')))
        info['recon'] = True


        prefix += '_inject_' + str(inject)
        prefix += '_start_layer_index_' + str(start_layer_index)
        prefix += '_end_layer_index_' + str(end_layer_index)

        if not os.path.exists(args.feature_path):
            os.mkdir(args.feature_path)
        with torch.no_grad():
            inp = prepare(t5, clip, init_image, prompt=opts.source_prompt)
            inp_target = prepare(t5, clip, init_image, prompt=opts.target_prompt)
            timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))

        # offload TEs to CPU, load model to gpu
        if offload:
            t5, clip = t5.cpu(), clip.cpu()
            torch.cuda.empty_cache()
            model = model.to(torch_device)

        denoise_strategies = {
            'reflow' : denoise,
            'reflow_m': denoise_m,
            'rf_solver' : denoise_rf_solver,
            'fireflow' : denoise_fireflow,
            'rf_midpoint' : denoise_midpoint,
            'fireflow_m': denoise_fireflow_M,
        }
        if args.sampling_strategy not in denoise_strategies:
            raise ExceptionType("Unknown denoising strategy")
        denoise_strategy = denoise_strategies[args.sampling_strategy]

        # inversion initial noise
        z, info = denoise_strategy(model, **inp, timesteps=timesteps, guidance=1, inverse=True, info=info)
        inp_target["img"] = z

        timesteps = get_schedule(opts.num_steps, inp_target["img"].shape[1], shift=(name != "flux-schnell"))

        # denoise initial noise
        x, _ = denoise_strategy(model, **inp_target, timesteps=timesteps, guidance=guidance, inverse=False, info=info)

        if offload:
            model.cpu()
            torch.cuda.empty_cache()
            ae.decoder.to(x.device)

        # decode latents to pixel space
        batch_x = unpack(x.float(), opts.width, opts.height)

        for x in batch_x:
            x = x.unsqueeze(0)
            output_name = os.path.join(output_dir, prefix + "_img_{idx}.jpg")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                idx = 0
            else:
                fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
                if len(fns) > 0:
                    idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
                else:
                    idx = 0

            with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                x = ae.decode(x)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            fn = output_name.format(idx=idx)
            print(f"Done in {t1 - t0:.1f}s. Saving {fn}")
            # bring into PIL format and save
            x = x.clamp(-1, 1)
            x = embed_watermark(x.float())
            x = rearrange(x[0], "c h w -> h w c")

            img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
            nsfw_score = [x["score"] for x in nsfw_classifier(img) if x["label"] == "nsfw"][0]

            if nsfw_score < NSFW_THRESHOLD:
                exif_data = Image.Exif()
                exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
                exif_data[ExifTags.Base.Make] = "Black Forest Labs"
                exif_data[ExifTags.Base.Model] = name
                if add_sampling_metadata:
                    exif_data[ExifTags.Base.ImageDescription] = source_prompt
                img.save(fn, exif=exif_data, quality=95, subsampling=0)
                idx += 1
            else:
                print("Your generated image may contain NSFW content.")

            if loop:
                print("-" * 80)
                opts = parse_prompt(opts)
            else:
                opts = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PMI')

    parser.add_argument('--name', default='flux-dev', type=str,
                        help='flux model')
    parser.add_argument('--source_img_dir', default='examples/source/red.jpg', type=str,
                        help='The path of the source image')
    parser.add_argument('--source_prompt', type=str,
                        default='Reflection of red Tibetan prayer flags and people on stone pavement.',
                        help='describe the content of the source image (or leaves it as null)')
    parser.add_argument('--target_prompt', type=str,
                        default='Reflection of red Tibetan prayer flags and people on stone pavement.',
                        help='describe the requirement of editing')
    parser.add_argument('--feature_path', type=str, default='feature',
                        help='the path to save the feature ')
    parser.add_argument('--guidance', type=float, default=1,
                        help='guidance scale')
    parser.add_argument('--num_steps', type=int, default=10,
                        help='the number of timesteps for inversion and denoising')
    parser.add_argument('--inject', type=int, default=1,
                        help='the number of timesteps which apply the feature sharing')
    parser.add_argument('--start_layer_index', type=int, default=20,
                        help='the number of block which starts to apply the feature sharing')
    parser.add_argument('--end_layer_index', type=int, default=37,
                        help='the number of block which ends to apply the feature sharing')
    parser.add_argument('--output_dir', default='examples/edit-result/recon/red', type=str,
                        help='the path of the edited image')
    parser.add_argument('--output_prefix', default='fireflow', type=str,
                        help='prefix name of the edited image')
    parser.add_argument('--sampling_strategy', default='fireflow', type=str,
                        help='method used to conduct sampling at inference time')
    parser.add_argument('--offload', action='store_true', help='set it to True if the memory of GPU is not enough')
    parser.add_argument('--reuse_v', type=int, default=1,
                        help='reuse v during inversion and reconstruction/editing')
    parser.add_argument('--editing_strategy', default='replace_v', type=str,
                        help='strategy for editing')
    parser.add_argument('--qkv_ratio', type=str, default='1.0,1.0,1.0',
                        help='A string of comma-separated float numbers')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')

    args = parser.parse_args()

    main(args)