import os
from PIL import Image
from pathlib import Path


# ============ Configuration Parameters ============
# These should match the values in scenario_drawer.py

# VIEW_SIZE_DEFAULT from scenario_drawer.py (in meters)
# This represents the full view range: -VIEW_SIZE/2 to +VIEW_SIZE/2
VIEW_SIZE_DEFAULT = 140.0

# ============ User Settings ============
# Input directory containing scenario subdirectories
INPUT_DIR = "/home/wenguang/workplace/fiss_plus_planner/fiss_plus_planner/output/generated_images"

# Output directory (folder structure will be preserved)
OUTPUT_DIR = "/home/wenguang/workplace/fiss_plus_planner/fiss_plus_planner/output/cropped_images"

# Crop range in meters (the range you want to keep, e.g., 70 means ±35m from center)
# Must be smaller than VIEW_SIZE_DEFAULT (140.0) to actually crop the image
# Examples:
#   - 140.0: no crop (same as original)
#   - 70.0:  keep ±35m, crop to half of original
#   - 50.0:  keep ±25m
CROP_RANGE_METERS = 70.0

# Output image size (square, in pixels)
OUTPUT_SIZE = 128
# =======================================


def calculate_crop_distance(crop_range_meters: float, image_size: int) -> int:
    """
    Calculate crop distance in pixels from crop range in meters.
    Uses actual image size to calculate the correct pixel distance.
    
    Args:
        crop_range_meters: The total range to keep in meters (e.g., 70 means ±35m from center)
        image_size: Actual image size in pixels
        
    Returns:
        Crop distance in pixels (half of crop box size)
    """
    # Calculate pixels per meter based on actual image size
    pixels_per_meter = image_size / VIEW_SIZE_DEFAULT
    # crop_range_meters is the total range, so we divide by 2 to get distance from center
    half_range_meters = crop_range_meters / 2.0
    return int(half_range_meters * pixels_per_meter)


def crop_and_resize_image(
    image_path: str,
    crop_range_meters: float,
    output_size: int,
    output_path: str
) -> None:
    """
    Crop an image from center and resize it.
    """
    with Image.open(image_path) as img:
        width, height = img.size
        
        # Calculate crop distance based on actual image size
        crop_distance = calculate_crop_distance(crop_range_meters, width)
        
        # Debug: print actual image size
        print(f"  Image: {width}x{height}, crop_distance: {crop_distance} pixels")
        
        # Calculate center point
        center_x = width // 2
        center_y = height // 2
        
        # Calculate crop box (left, upper, right, lower)
        left = max(0, center_x - crop_distance)
        upper = max(0, center_y - crop_distance)
        right = min(width, center_x + crop_distance)
        lower = min(height, center_y + crop_distance)
        
        # Crop the image
        cropped_img = img.crop((left, upper, right, lower))
        
        # Resize to output size
        resized_img = cropped_img.resize((output_size, output_size), Image.LANCZOS)
        
        # Create output directory if not exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the image
        resized_img.save(output_path)
        print(f"  Saved: {output_path}")


def process_directory(
    input_dir: str,
    output_dir: str,
    crop_range_meters: float,
    output_size: int
) -> None:
    """
    Process all PNG images in subdirectories, preserving folder structure.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    # Create output base directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Traverse all subdirectories (scenarios)
    for scenario_dir in input_path.iterdir():
        if scenario_dir.is_dir():
            print(f"\nProcessing scenario: {scenario_dir.name}")
            
            # Create corresponding output directory
            scenario_output_dir = output_path / scenario_dir.name
            os.makedirs(scenario_output_dir, exist_ok=True)
            
            # Find all PNG files in the scenario directory
            png_files = list(scenario_dir.glob("*.png"))
            print(f"Found {len(png_files)} PNG files")
            
            for png_file in png_files:
                try:
                    # Preserve original filename in output
                    output_file_path = scenario_output_dir / png_file.name
                    crop_and_resize_image(
                        str(png_file),
                        crop_range_meters,
                        output_size,
                        str(output_file_path)
                    )
                except Exception as e:
                    print(f"Error processing {png_file}: {e}")


def main():
    print("=" * 60)
    print("Image Crop and Resize Tool")
    print("=" * 60)
    print(f"Input directory:  {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 60)
    print(f"VIEW_SIZE_DEFAULT (from scenario_drawer): {VIEW_SIZE_DEFAULT} meters")
    print(f"Crop range: {CROP_RANGE_METERS} meters (±{CROP_RANGE_METERS/2} meters from center)")
    print(f"Output size: {OUTPUT_SIZE}x{OUTPUT_SIZE} pixels")
    print("=" * 60)
    print("Note: Original images will NOT be modified.")
    print("=" * 60)
    
    process_directory(
        INPUT_DIR,
        OUTPUT_DIR,
        CROP_RANGE_METERS,
        OUTPUT_SIZE
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
