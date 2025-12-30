"""
This script provides a utility to resize images in a directory to a specified
target size while maintaining their aspect ratio. It uses a thread pool for
concurrent processing to improve performance.
"""
import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def process_image(filename, directory, output_directory, target_size):
    """
    Processes a single image: resizes it if necessary and saves it to the output directory.

    If the image's longest side is larger than `target_size`, it is resized.
    Otherwise, it is copied directly.

    Args:
        filename (str): The name of the image file.
        directory (str): The directory where the image is located.
        output_directory (str): The directory to save the processed image.
        target_size (int): The target size for the longest side of the image.

    Returns:
        int: 1 if the image was processed successfully, 0 otherwise.
    """
    try:
        with Image.open(os.path.join(directory, filename)) as img:
            width, height = img.size
            print(f'Processing Image: {filename} | Resolution: {width}x{height}')

            # Determine the scaling ratio to make the longest side equal to target_size.
            if max(width, height) > target_size:
                if width > height:
                    new_width = target_size
                    new_height = int((target_size / width) * height)
                else:
                    new_height = target_size
                    new_width = int((target_size / height) * width)

                # Resize the image using the ANTIALIAS filter for high-quality downsampling.
                img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Save the resized image to the output directory.
                output_path = os.path.join(output_directory, filename)
                img_resized.save(output_path)
                print(f'Resized and saved {filename} to {output_directory} with resolution {new_width}x{new_height}')
            else:
                # If the image does not need resizing, save it directly to the output directory.
                img.save(os.path.join(output_directory, filename))
                print(f'Image {filename} already meets the target size and was saved without resizing.')
            return 1  # Indicate successful processing.
    except Exception as e:
        print(f"Cannot process {filename}: {e}")
        return 0  # Indicate processing failure.

def get_image_resolutions_and_resize(directory='.', output_directory='resized_images', target_size=1024):
    """
    Finds all images in a directory, resizes them concurrently, and saves them to an output directory.

    Args:
        directory (str): The directory containing the images to process.
        output_directory (str): The directory where resized images will be saved.
        target_size (int): The target size for the longest side of the images.
    """
    # Create the output directory if it does not exist.
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Get a list of all image files in the source directory.
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff'))]
    
    # Use a thread pool to process images concurrently for better performance.
    total_processed = 0
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, filename, directory, output_directory, target_size) for filename in image_files]
        
        # Wait for all threads to complete and aggregate the count of processed images.
        for future in futures:
            total_processed += future.result()

    # Print the total number of images processed.
    print(f"\nTotal number of images processed: {total_processed}")

if __name__ == "__main__":
    # --- Configuration ---
    # Set the source and destination directories, and the target size.
    get_image_resolutions_and_resize(
        directory="./compRAISE",
        output_directory="./compRAISE1024",
        target_size=1024
    )
