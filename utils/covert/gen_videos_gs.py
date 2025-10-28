from PIL import Image, ImageDraw
from pathlib import Path
import argparse

def draw_red_box(image, thickness=10):
    """
    Draws a red border around the given image.
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for i in range(thickness):
        draw.rectangle(
            [(i, i), (width - 1 - i, height - 1 - i)],
            outline="red"
        )
    return image

def stitch_images(
    source_path_template, 
    base_path_template, 
    output_path, 
    mode, 
    indices, 
    columns=8, 
    use_source=True
):
    """
    Stitches images based on the specified mode (include/exclude) and source usage.
    """
    indices_to_highlight = set(indices)
    
    # 1. Collect indices to process
    print(f"1. Current mode: '{mode}'. Preparing image indices...")
    indices_to_process = []
    if mode == 'exclude':
        all_indices = {f"{i:02d}" for i in range(48)}
        indices_to_process = sorted(list(all_indices - indices_to_highlight))
    elif mode == 'include':
        indices_to_process = sorted(list(indices_to_highlight))
    else:
        print(f"Error: Unknown mode '{mode}'. Please use 'include' or 'exclude'.")
        return

    print(f"   Will process {len(indices_to_process)} indices.")

    # 2. Validate paths and get image dimensions
    print("2. Validating paths and getting image dimensions...")
    first_valid_image = None
    if indices_to_process:
        first_img_path = Path(base_path_template.format(index=indices_to_process[0]))
        if first_img_path.exists():
            with Image.open(first_img_path) as img:
                first_valid_image = img.copy()
    
    if not first_valid_image:
        print("Error: No valid image files found in base_path. Please check the path.")
        return

    img_width, img_height = first_valid_image.size
    
    # 3. Calculate final canvas size
    total_indices = len(indices_to_process)
    num_image_rows = (total_indices + columns - 1) // columns

    if use_source:
        num_canvas_rows = num_image_rows * 2
        print("   Mode: Alternating stitch (source + base).")
    else:
        num_canvas_rows = num_image_rows
        print("   Mode: Single source stitch (base only).")

    canvas_width = img_width * columns
    canvas_height = img_height * num_canvas_rows
    
    print(f"   Single image size: {img_width}x{img_height}")
    print(f"   Final canvas size: {canvas_width}x{canvas_height} ({columns} columns x {num_canvas_rows} rows)")

    canvas = Image.new('RGB', (canvas_width, canvas_height))

    print("4. Stitching images...")
    
    for i, index_str in enumerate(indices_to_process):
        row = i // columns
        col = i % columns
        paste_x = col * img_width
        
        if use_source:
            # --- Alternating Mode ---
            source_y = row * 2 * img_height
            source_img_path = Path(source_path_template.format(index=index_str))
            if source_img_path.exists():
                with Image.open(source_img_path) as img:
                    img_to_paste = img.convert('RGB')
                    if index_str in indices_to_highlight:
                        img_to_paste = draw_red_box(img_to_paste)
                    canvas.paste(img_to_paste, (paste_x, source_y))

            base_y = (row * 2 + 1) * img_height
            base_img_path = Path(base_path_template.format(index=index_str))
            if base_img_path.exists():
                with Image.open(base_img_path) as img:
                    img_to_paste = img.convert('RGB')
                    if index_str in indices_to_highlight:
                        img_to_paste = draw_red_box(img_to_paste)
                    # --- THIS IS THE CORRECTED LINE ---
                    canvas.paste(img_to_paste, (paste_x, base_y))
        else:
            # --- Single Source Mode ---
            paste_y = row * img_height
            base_img_path = Path(base_path_template.format(index=index_str))
            if base_img_path.exists():
                with Image.open(base_img_path) as img:
                    img_to_paste = img.convert('RGB')
                    if index_str in indices_to_highlight:
                        img_to_paste = draw_red_box(img_to_paste)
                    canvas.paste(img_to_paste, (paste_x, paste_y))

        if (i + 1) % columns == 0:
            print(f"   Completed stitching for image group {row + 1}...")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_file)
    
    print(f"\n5. Stitching complete! Image saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Stitch images from two sources, with highlighting.")
    
    parser.add_argument("--source_path", type=str, 
                        default="/mnt/cvda/cvda_phava/code/Han/Diffuman4D/data/datasets--krahets--diffuman4d_example/0023_06/images/{index}/000000.webp",
                        help="Source image path template (.webp)")
                        
    parser.add_argument("--base_path", type=str, 
                        default="/mnt/cvda/cvda_phava/code/Han/Diffuman4D/output/results/demo_3d/0023_06/images/{index}/000000.jpg",
                        help="Base image path template (.jpg)")
    
    parser.add_argument("--output", "-o", type=str, 
                        default="/mnt/cvda/cvda_phava/code/Han/Diffuman4D/output/results/demo_3d/0023_06/stitched_result.jpg",
                        help="Output path for the combined image.")
                        
    parser.add_argument("--columns", type=int, default=8, help="Number of images per row.")

    parser.add_argument("--mode", type=str, default="include", choices=["include", "exclude"],
                        help="Operation mode: 'exclude' or 'include' the specified indices.")
                        
    parser.add_argument("--indices", type=str, nargs='+', default=["01", "13", "25", "37"],
                        help="List of indices to highlight (and filter by).")
    
    parser.add_argument("--no_source", action="store_true",
                        help="If set, only stitch images from base_path.")

    args = parser.parse_args()

    output_path = Path(args.output)
    suffix = "_no_source" if args.no_source else "_alternating"
    new_filename = f"{output_path.stem}_{args.mode}{suffix}{output_path.suffix}"
    final_output_path = output_path.with_name(new_filename)

    stitch_images(
        args.source_path, 
        args.base_path, 
        final_output_path, 
        args.mode, 
        args.indices, 
        args.columns,
        use_source=not args.no_source
    )


if __name__ == "__main__":
    main()