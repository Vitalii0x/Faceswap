## Face Swap Toolkit — Usage Guide

This repository provides scripts to perform one-shot face swapping on images and videos, plus optional super-resolution and training code.

### Requirements
- Python 3.7+
- NVIDIA GPU with CUDA recommended
- ffmpeg available in your PATH (required for adding audio back to output videos)

Install Python dependencies:
```bash
pip install -r requirements.txt
```
If `onnxruntime-gpu` is unavailable for your platform, install `onnxruntime` instead. If your CUDA version differs from the one pinned in `requirements.txt`, install a matching `torch`/`torchvision` build for your system.

### Download pretrained models
From a Bash shell (Git Bash/WSL/macOS/Linux):
```bash
bash download_models.sh
```
This populates `weights/`, `arcface_model/`, and `insightface_func/models/antelope/` with required files. Alternatively, download the URLs listed in `download_models.sh` manually.

### Quick start
Run face swap on a video, specifying a source face image and one or more target face crops (faces from the target video to be replaced):
```bash
python inference.py \
  --source_paths examples/images/elon_musk.jpg \
  --target_faces_paths examples/images/beckham.jpg \
  --target_video examples/videos/dance.mp4 \
  --out_video_name examples/results/result.mp4
```

Swap multiple faces (order of `--source_paths` must match `--target_faces_paths`):
```bash
python inference.py \
  --source_paths img1.jpg img2.jpg \
  --target_faces_paths tgt_face1.jpg tgt_face2.jpg \
  --target_video input.mp4
```

Image-to-image swap:
```bash
python inference.py \
  --source_paths examples/images/elon_musk.jpg \
  --image_to_image True \
  --target_image examples/images/beckham.jpg \
  --out_image_name examples/results/result.png
```

Enable optional face super-resolution on generated faces:
```bash
python inference.py ... --use_sr True
```

### Inference arguments (key options)
- `--source_paths`: one or more source face images
- `--target_faces_paths`: one or more face crops from the target video to replace (optional; if omitted, the most similar face is selected automatically)
- `--target_video`: path to target video (for video mode)
- `--out_video_name`: output video path
- `--image_to_image`: set `True` for image-to-image mode
- `--target_image`: path to target image (for image mode)
- `--out_image_name`: output image path
- `--G_path`: generator weights path (default `weights/G_unet_2blocks.pth`)
- `--backbone`: `unet` | `linknet` | `resnet`
- `--num_blocks`: number of AAD blocks in the generator (default 2)
- `--batch_size`: batch size for inference (default 40)
- `--use_sr`: enable face super-resolution
- `--similarity_th`: threshold for face selection in video

Do not change `--crop_size` (fixed at 224).

### Training
1) Preprocess dataset (e.g., VGGFace2) by detecting and aligning faces:
```bash
python preprocess_vgg.py --path_to_dataset /path/to/raw_dataset --save_path /path/to/aligned_dataset
```

2) Start training:
```bash
python train.py --run_name my_experiment
```
Common options:
- `--vgg True|False`: set to `True` for multi-image-per-identity datasets (e.g., VGGFace2)
- `--eye_detector_loss True|False`: add eye landmark loss
- `--scheduler True|False`: use learning-rate scheduler
- `--pretrained True|False`: start from provided weights in `weights/`

Training outputs are saved under `saved_models_<run_name>/` and `current_models_<run_name>/`.

### Notes
- ffmpeg is required to reattach audio to the output video.
- To use super-resolution (`--use_sr True`), ensure `weights/10_net_G.pth` is downloaded.
- Example assets are under `examples/images` and `examples/videos`.
