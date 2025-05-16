import os
import argparse
from utils.gpt4o import gpt4o_response
from tqdm import tqdm
import concurrent.futures

# Fixed prompt template
prompt_template = """This is an AI-generated image, tell me the high-level reasons you observed that support this conclusion in detail. High-level errors are semantic or structural issues that affect the overall logic, coherence, or realism of the image. These errors are typically noticeable even without close inspection and relate to the broader understanding of the scene or objects.

1. Here are some high level errors you can consider:

    1. Human Anatomy
        (1) Hands and fingers
        - Extra or missing fingers
        - Unnatural proportion
        - Irregular structure

        (2) Posture and movement inconsistencies
        - Unnaturally positioned joints (like elbows, knees, shoulders, wrists or ankles)
        - Contradict directions of body and legs
        - Impossible posture or guesture

        (3) Facial features
        - Facial features uneven in size, shape, or position (like mouth, ears or eyes)
        - Missing details (like eyebrows, eyelashes, or lip texture may be absent or overly simplified)

        (4) Arms and legs
        - Redundant / missing arm or leg
        - Oversized or elongated structure

    
    2. Semantic Errors:
        (1) Unreadable or distorted text
        (2) Unrealistic object interactions (unnatural overlaps or impossible spatial arrangements)
        (3) Violation of physical laws (floating objects etc.)
        (4) Contextual inconsistencies (unconventional environment setting)
        (5) Visual inconsistencies (objects that do not align with real world)


    3. Lighting and Shadow Errors:
        (1) Shadow size mismatch (too large or too small)
        (2) Shadow position mismatch (incorrect direction relative to light source)
        (3) Missing shadow in a scene that requires it
        (4) Inconsistent reflections

    4. Others:
        (1) Object distortion 

2. Do not list any low-level error in your answer. Low-level errors are more subtle and relate to fine details, textures, or visual artifacts that may not be immediately obvious without closer inspection.
3. If the image is in a non-realistic style, such as a cartoon or painting, you should point it out.
4. Do not list any related high-level error if such error or related element does not exist in this image.
5. Analyze the image point by point according to the given key points to check if there are errors corresponding to each key point.
6. You should give a **detailed** location and explanation in every reason you give.
7. Place each of your points between `<begin_of_point>` and `<end_of_point>`.
8. If you do not observe any high-level errors, conclude that in the end of your response."""




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
    parser.add_argument("--output_folder", default="generated_annotation_high_level_norefined", help="Path to the folder where output text files will be saved.")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of worker threads. (Default: number of CPU cores)")
    args = parser.parse_args()

    process_images_parallel(args.input_folder, args.output_folder, prompt_template, args.max_workers)

if __name__ == "__main__":
    main()