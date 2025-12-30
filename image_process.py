"""
This script provides utilities for collecting image file paths and generating
JSON datasets for image manipulation detection tasks. It pairs tampered images
with their corresponding ground truth masks and can optionally include authentic
images.
"""
import os
import json

def collect_image_paths(root_dir):
    """
    Recursively collects all image file paths from a given directory.

    Args:
        root_dir (str): The root directory to search for images.

    Returns:
        list[str]: A list of relative paths to the found image files.
    """
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    image_paths = []
    
    # Normalize the root directory to an absolute path.
    root_dir = os.path.abspath(root_dir)
    
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in image_exts:
                # Create the full file path.
                full_path = os.path.join(dirpath, filename)
                # Convert to a path relative to the current working directory.
                rel_path = os.path.relpath(full_path, start=os.getcwd())
                # Ensure consistent path separators (forward slashes).
                rel_path = rel_path.replace('\\', '/')
                image_paths.append(rel_path)
    return image_paths

def generate_pairs(image_root, mask_root, output_json, authentic_root=None, include_authentic=True):
    """
    Generates pairs of tampered images and their masks, and optionally includes authentic images.

    The output is a JSON file containing a list of pairs. For tampered images,
    a pair is [image_path, mask_path]. For authentic images, it is [image_path, "Negative"].

    Args:
        image_root (str): The root directory of the tampered images.
        mask_root (str): The root directory of the ground truth masks.
        output_json (str): The path to the output JSON file.
        authentic_root (str, optional): The root directory of the authentic images. Defaults to None.
        include_authentic (bool): Whether to include authentic images in the dataset. Defaults to True.
    
    Returns:
        tuple[list, list]: A tuple containing the list of tampered pairs and the list of authentic pairs.
    """
    all_pairs = []
    
    # Process tampered image pairs.
    print("Processing tampered image pairs...")
    image_dict = collect_image_paths(image_root)
    print(f"Found {len(image_dict)} tampered images.")
    mask_dict = collect_image_paths(mask_root)
    print(f"Found {len(mask_dict)} masks.")
    assert len(image_dict) == len(mask_dict), \
        f"The number of images ({len(image_dict)}) and masks ({len(mask_dict)}) do not match!"
    
    # Create a sorted list of [image, mask] pairs.
    tampered_pairs = [
        list(pairs)
        for pairs in zip(sorted(image_dict), sorted(mask_dict))
    ]
    all_pairs.extend(tampered_pairs)
    print(f"Successfully generated {len(tampered_pairs)} tampered image pairs.")
    
    # Process authentic images.
    authentic_pairs = []
    if include_authentic and authentic_root and os.path.exists(authentic_root):
        print("\nProcessing authentic images...")
        authentic_images = collect_image_paths(authentic_root)
        print(f"Found {len(authentic_images)} authentic images.")
        
        # Create pairs for authentic images, marking them as "Negative".
        authentic_pairs = [
            [image_path, "Negative"]
            for image_path in sorted(authentic_images)
        ]
        all_pairs.extend(authentic_pairs)
        print(f"Successfully generated {len(authentic_pairs)} authentic image records.")
    elif include_authentic and authentic_root:
        print(f"Warning: Authentic image directory '{authentic_root}' not found. Skipping authentic images.")
    
    # Save the combined list of pairs to a JSON file.
    with open(output_json, 'w') as f:
        json.dump(all_pairs, f, indent=2)

    print(f"\nTotal generated {len(all_pairs)} records ({len(tampered_pairs)} tampered + {len(authentic_pairs)} authentic). "
          f"Results saved to {output_json}")
    
    return tampered_pairs, authentic_pairs

if __name__ == "__main__":
    # --- Configuration ---
    # Modify these paths according to your dataset structure.
    IMAGE_ROOT = "Dataset/CASIA1.0/Tp"      # Root directory for tampered images.
    MASK_ROOT = "Dataset/CASIA1.0/Gt"       # Root directory for ground truth masks.
    AUTHENTIC_ROOT = "Dataset/CASIA1.0/Au"  # Root directory for authentic images.
    OUTPUT_JSON = "CASIAv1.json"            # Name of the output JSON file.
    INCLUDE_AUTHENTIC = True                # Set to False to exclude authentic images.

    # --- Execution ---
    tampered_pairs, authentic_pairs = generate_pairs(
        IMAGE_ROOT, 
        MASK_ROOT, 
        OUTPUT_JSON, 
        AUTHENTIC_ROOT, 
        INCLUDE_AUTHENTIC
    )

    # --- Verification ---
    # Print the last 5 tampered pairs to verify alignment.
    print("\nExample of the last 5 tampered image pairs:")
    for pair in tampered_pairs[-5:]:
        print(f"  Image: {pair[0]}")
        print(f"  Mask:  {pair[1]}\n")
    
    if authentic_pairs:
        print(f"Additionally, {len(authentic_pairs)} authentic image records were included.")
