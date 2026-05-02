import argparse
import glob
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch

from core.ma_raft import MA_RAFT


IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")


def read_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")

    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    return image


def robust_normalize(image: np.ndarray, lower: float, upper: float) -> np.ndarray:
    image = image.astype(np.float32)
    vmin, vmax = np.percentile(image, [lower, upper])
    if vmax - vmin < 1e-5:
        return np.zeros_like(image, dtype=np.uint8)

    image = (image - vmin) * 255.0 / (vmax - vmin)
    return np.clip(image, 0, 255).astype(np.uint8)


def image_to_tensor(image: np.ndarray, device: torch.device, size: int | None) -> torch.Tensor:
    if size is not None and image.shape[:2] != (size, size):
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return torch.from_numpy(image).permute(2, 0, 1).float()[None].to(device)


def write_flo(path: Path, flow: np.ndarray) -> None:
    h, w, _ = flow.shape
    with path.open("wb") as file:
        np.array([202021.25], dtype=np.float32).tofile(file)
        np.array([w], dtype=np.int32).tofile(file)
        np.array([h], dtype=np.int32).tofile(file)
        flow.astype(np.float32).tofile(file)


def list_images(data_dir: Path) -> list[Path]:
    images: list[str] = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(glob.glob(str(data_dir / ext)))
    return [Path(path) for path in sorted(images)]


def build_model(checkpoint: Path, device: torch.device, mixed_precision: bool) -> MA_RAFT:
    args = argparse.Namespace(dropout=0.0, mixed_precision=mixed_precision)
    model = MA_RAFT(args).to(device)

    # Build lazy correlation parameters before loading checkpoints saved after a warmup.
    dummy = torch.zeros(1, 3, 256, 256, device=device)
    with torch.no_grad():
        model(dummy, dummy, iters=1, test_mode=True)

    state = torch.load(checkpoint, map_location=device)
    state = state.get("state_dict", state)
    clean_state = OrderedDict((k.removeprefix("module."), v) for k, v in state.items())
    model.load_state_dict(clean_state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def predict(args: argparse.Namespace) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_model(args.model, device, args.mixed_precision)

    images = list_images(args.data)
    if len(images) < 2:
        raise ValueError(f"Need at least two images in {args.data}")
    if args.frame_index < 0 or args.frame_index + 1 >= len(images):
        raise IndexError(f"frame_index must be in [0, {len(images) - 2}]")

    img1 = robust_normalize(read_grayscale(images[args.frame_index]), args.norm_lower, args.norm_upper)
    img2 = robust_normalize(read_grayscale(images[args.frame_index + 1]), args.norm_lower, args.norm_upper)
    ten1 = image_to_tensor(img1, device, args.size)
    ten2 = image_to_tensor(img2, device, args.size)

    _, flow = model(ten1, ten2, iters=args.iters, test_mode=True)
    flow_np = flow[0].permute(1, 2, 0).cpu().numpy()

    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / f"result_flow_frame_{args.frame_index}.flo"
    write_flo(output_path, flow_np)

    speed = np.linalg.norm(flow_np, axis=2)
    print(f"Saved: {output_path}")
    print(f"Frames: {images[args.frame_index].name} -> {images[args.frame_index + 1].name}")
    print(f"Mean speed: {speed.mean():.4f} px, max speed: {speed.max():.4f} px")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MA_RAFT on one adjacent image pair.")
    parser.add_argument("--model", type=Path, required=True, help="Path to MA_RAFT checkpoint.")
    parser.add_argument("--data", type=Path, required=True, help="Directory containing image sequence.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for .flo output.")
    parser.add_argument("--frame-index", type=int, default=0, help="First frame index.")
    parser.add_argument("--iters", type=int, default=16, help="RAFT update iterations.")
    parser.add_argument("--size", type=int, default=256, help="Square resize size; use 0 to keep original size.")
    parser.add_argument("--norm-lower", type=float, default=0.0, help="Lower percentile for normalization.")
    parser.add_argument("--norm-upper", type=float, default=100.0, help="Upper percentile for normalization.")
    parser.add_argument("--mixed-precision", action="store_true", help="Use CUDA autocast during inference.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    args = parser.parse_args()
    args.size = None if args.size == 0 else args.size
    return args


if __name__ == "__main__":
    predict(parse_args())
