import argparse
import math
import random
from pathlib import Path

from PIL import Image


def collect_patches(input_dir: Path, class_filter: str | None, limit: int | None) -> list[Path]:
    if input_dir.is_file():
        return [input_dir]

    patches = []
    for path in input_dir.rglob("*.png"):
        if class_filter and not path.parent.name.lower().startswith(class_filter.lower()):
            continue
        patches.append(path)

    if not patches:
        raise FileNotFoundError("No patches found with the provided filters")

    random.shuffle(patches)
    if limit:
        patches = patches[:limit]

    return patches


def build_mosaic(patches: list[Path], grid_rows: int, grid_cols: int, scale: float) -> Image.Image:
    first = Image.open(patches[0])
    patch_w, patch_h = first.size
    first.close()

    width = int(patch_w * grid_cols * scale)
    height = int(patch_h * grid_rows * scale)

    mosaic = Image.new("RGB", (width, height))

    for idx, patch_path in enumerate(patches):
        if idx >= grid_rows * grid_cols:
            break

        with Image.open(patch_path) as patch:
            if scale != 1.0:
                patch = patch.resize((int(patch_w * scale), int(patch_h * scale)), Image.Resampling.NEAREST)
            row = idx // grid_cols
            col = idx % grid_cols
            mosaic.paste(patch, (col * patch.width, row * patch.height))

    return mosaic


def auto_grid(num_patches: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(num_patches))
    rows = math.ceil(num_patches / cols)
    return rows, cols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine HistAerial patches into a larger mosaic")
    parser.add_argument("input", type=str, help="Directory containing patches or a single patch file")
    parser.add_argument("output", type=str, help="Output image path (e.g., mosaic.png)")
    parser.add_argument("--num_patches", type=int, default=25, help="Number of patches to include")
    parser.add_argument("--grid", type=str, default=None, help="Grid layout rowsxcols (e.g., 5x5). Defaults to automatic square grid")
    parser.add_argument("--class_prefix", type=str, default=None, help="Optional class prefix filter (e.g., URBAINS, PRAIRIES)")
    parser.add_argument("--scale", type=float, default=2.0, help="Upscale factor for each patch (default: 2.0)")

    args = parser.parse_args()

    if args.grid:
        try:
            rows_str, cols_str = args.grid.lower().split("x")
            args.grid_rows = int(rows_str)
            args.grid_cols = int(cols_str)
        except ValueError as exc:
            raise ValueError("--grid must be in the form rowsxcols, e.g., 5x5") from exc
    else:
        args.grid_rows = args.grid_cols = None

    return args


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input)
    patches = collect_patches(input_dir, args.class_prefix, args.num_patches)

    if args.grid_rows is None or args.grid_cols is None:
        grid_rows, grid_cols = auto_grid(len(patches))
    else:
        grid_rows, grid_cols = args.grid_rows, args.grid_cols

    mosaic = build_mosaic(patches, grid_rows, grid_cols, args.scale)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mosaic.save(output_path)
    print(f"Saved mosaic with {grid_rows}x{grid_cols} patches to {output_path}")


if __name__ == "__main__":
    main()
