import os
import argparse
from utils.gpt4o import gpt4o_response
from tqdm import tqdm
import concurrent.futures

# Fixed prompt template
prompt_template = """You are provided with two sets of annotations for a synthetic image. One set focuses on **low-level errors**, which are subtle issues related to fine details, textures, or visual artifacts that may not be immediately obvious without closer inspection. The other set focuses on **high-level errors**, which are semantic or structural issues affecting the overall logic, coherence, or realism of the image. These errors are typically noticeable even at a glance and relate to the broader understanding of the scene or objects.
Your task is to merge these two sets of annotations into a single, structured annotation following these rules:  
1. For each point, If the point describes an error, use the format: "**Error Type**: Detailed description". Otherwise, restate the point semantically as it is. Enclose each point between `<begin_of_point>` and `<end_of_point>`. Do not number the points.
2. Place all low-level error points between `<begin_of_low_level_errors>` and `<end_of_low_level_errors>`, and all high-level error points between `<begin_of_high_level_errors>` and `<end_of_high_level_errors>`. Arrange the points in the order they appear.
3. Since the low-level error annotations may include high-level errors, and the high-level error annotations may include low-level errors, carefully reclassify and merge semantically overlapping points. Ensure that the final annotation correctly categorizes errors into low-level and high-level groups without redundancy.
4. At the end of the annotation, provide a conclusion stating whether the image is synthetic or real. Place your judgment label inside `\\boxed{{}}`, using either `real` or `AI-generated`.

**Example Annotation Format**:

<begin_of_low_level_errors>
<begin_of_point>
**Texture Inconsistency**: The texture of the wooden table appears overly smooth in some areas, lacking the fine grain details expected in realistic wood surfaces.  
<end_of_point>
<begin_of_point>
**Lighting Artifact**: A faint halo effect is visible around the edges of the glass object, likely caused by improper blending of light reflections.  
<end_of_point>
<end_of_low_level_errors>

<begin_of_high_level_errors>
<begin_of_point>
**Object Proportion Mismatch**: The size of the book relative to the chair is unrealistic, as the book appears disproportionately large for its context.  
<end_of_point>
<begin_of_point>
**Scene Coherence Issue**: The placement of the lamp behind the chair creates an illogical shadow pattern, disrupting the overall realism of the scene.  
<end_of_point>
<end_of_high_level_errors>

**Conclusion**: Based on the combination of low-level and high-level errors identified, the image is judged to be \\boxed{{AI-generated}}.

Use this structure to process and merge the provided annotations while ensuring clarity, correctness, and adherence to the given guidelines.

### Low-level Error Annotation:

{low_level_annotation}

### High-level Error Annotation:

{high_level_annotation}"""




def process_image(image_path, low_level_folder, high_level_folder, output_folder, prompt_template):
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
    low_level_path = os.path.join(low_level_folder, subfolder, output_filename)
    high_level_path = os.path.join(high_level_folder, subfolder, output_filename)

    # Skip if the output file already exists
    if os.path.exists(output_path):
        print(f"Skipping {output_path}, output file already exists.")
        return  # Skip further processing
    
    if not os.path.exists(low_level_path):
        print(f"Missing low-level annotation: {low_level_path}, skipping...")
        return
    
    if not os.path.exists(high_level_path):
        print(f"Missing high-level annotation: {high_level_path}, skipping...")
        return
    
    with open(low_level_path, "r") as f:
        low_level_annotation = f.read()

    with open(high_level_path, "r") as f:
        high_level_annotation = f.read()

    prompt = prompt_template.format(low_level_annotation=low_level_annotation, high_level_annotation=high_level_annotation)

    # Call the GPT-4 API
    try:
        response = gpt4o_response(prompt, image_path)

        # Write the result to the output file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response)

        print(f"Processed {image_path}, result saved to {output_path}")

    except Exception as e:
        print(f"Error processing {output_path}: {e}")

def process_images_parallel(input_folder, low_level_folder, high_level_folder, output_folder, prompt_template, max_workers=None):

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
        list(tqdm(
            executor.map(process_image, image_paths,  
            [low_level_folder] * len(image_paths), 
            [high_level_folder] * len(image_paths), 
            [output_folder] * len(image_paths),
            [prompt_template] * len(image_paths)), 
            total=len(image_paths), 
            desc="Processing images"
            ))

def main():
    parser = argparse.ArgumentParser(description="Process images with GPT-4 and a prompt template (parallel processing).")
    parser.add_argument("--input_folder", default="generated_images", help="Path to the folder containing images.")
    parser.add_argument("--low_level_folder", default="generated_annotation_low_level", help="Path to the folder where low-level output text files will be saved.")
    parser.add_argument("--high_level_folder", default="generated_annotation_high_level_refined", help="Path to the folder where high-level output text files will be saved.")
    parser.add_argument("--output_folder", default="generated_annotation_final", help="Path to the folder where final output text files will be saved.")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of worker threads. (Default: number of CPU cores)")
    args = parser.parse_args()

    process_images_parallel(args.input_folder, args.low_level_folder, args.high_level_folder, args.output_folder, prompt_template, args.max_workers)

if __name__ == "__main__":
    main()