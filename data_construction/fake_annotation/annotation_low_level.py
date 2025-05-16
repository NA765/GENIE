import os
import argparse
from utils.gpt4o import gpt4o_response
from tqdm import tqdm
import concurrent.futures

# Fixed prompt template
prompt_template = """This is an AI-generated image, please only list the most obvious low-level errors you observed in this image. Low-level errors are more subtle and relate to fine details, textures, or visual artifacts that may not be immediately obvious without closer inspection. Do not list too many low-level errors."""




def process_image(image_path, output_folder, prompt_template):
    """
    Process a single image, call the GPT-4 API, and save the result to a file.
    """
    filename = os.path.basename(image_path)
    output_filename = os.path.splitext(filename)[0] + ".txt"
    subfolder = os.path.basename(os.path.dirname(image_path))

    current_folder = os.path.join(output_folder, subfolder)
    if not os.path.exists(current_folder):
        os.makedirs(current_folder, exist_ok=True)

    output_path = os.path.join(current_folder, output_filename)

    # Skip if the output file already exists
    if os.path.exists(output_path):
        print(f"Skipping {output_path}, output file already exists.")
        return  # Skip further processing

    # Call the GPT-4 API
    try:
        response = gpt4o_response(prompt_template, image_path)

        # Write the result to the output file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response)

        print(f"Processed {image_path}, result saved to {output_path}")

    except Exception as e:
        print(f"Error processing {output_path}: {e}")

def process_images_parallel(input_folder, output_folder, prompt_template, max_workers=None):
    """
    Process images in parallel from the input folder.

    Args:
        input_folder: Path to the folder containing images.
        output_folder: Path to the folder where output text files will be saved.
        prompt_template: Prompt template for analysis.
        max_workers: Maximum number of worker threads. If None, defaults to the number of CPU cores.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Collect all image paths directly from the input folder
    image_paths = []
    for subfolder in os.listdir(input_folder):
        current_folder = os.path.join(input_folder, subfolder)
        for filename in os.listdir(current_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_paths.append(os.path.join(current_folder, filename))

    # Use a thread pool to process images in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(process_image, image_paths, [output_folder] * len(image_paths), [prompt_template] * len(image_paths)),
                  total=len(image_paths), desc="Processing images"))

def main():
    parser = argparse.ArgumentParser(description="Process images with GPT-4 and a prompt template (parallel processing).")
    parser.add_argument("--input_folder", default="generated_images", help="Path to the folder containing images.")
    parser.add_argument("--output_folder", default="generated_annotation_low_level", help="Path to the folder where output text files will be saved.")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of worker threads. (Default: number of CPU cores)")
    args = parser.parse_args()

    process_images_parallel(args.input_folder, args.output_folder, prompt_template, args.max_workers)

if __name__ == "__main__":
    main()