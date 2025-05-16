import json
import re
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from tqdm import tqdm

from utils.gpt4o import gpt4o_response



manual_revise_prompt = """You are tasked with revising and improving a piece of text that describes errors present in a synthetic image. The original annotation text may contain incomplete or unclear expressions that require adjustment. Using the manual revision hints provided, refine the original text to make it more accurate, clear, and comprehensive.  

**Original Annotation Text:**  
{original_text}

**Manual Revision Hints:**  
{manual_revisions}

Based on the hints, rewrite the original annotation text to better describe the errors in the synthetic image. Here are some tips you should follow:
1. Ensure that the revised text is **detailed**, concise, precise in each point, and fully incorporates the suggested changes. 
2. Keep placing each of your points between `<begin_of_point>` and `<end_of_point>`.
3. The listed errors can only be derived from the Original Annotation Text and Manual Revision Hints. It is **prohibited** to generate any other types of error.
4. If both the Original Annotation Text and Manual Revision Hints are empty, you should output a single point stating that the image appears realistic with no obvious high-level errors.
5. Here are some high-level error classifications you can consider:

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
        (1) Object distortion"""




def replace_image_path(url, image_root):
    # match any protocol/domain/IP, extract the path after images/
    pattern = r".*/images/(.*)"
    match = re.match(pattern, url)
    if match:
        local_path = f"{image_root}/{match.group(1)}"
        return local_path
    return url  # return original path if not matched

def extract_data(json_file, image_root):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    for task in data:
        # replace image path
        original_image_url = task['data']['image']
        local_image_path = replace_image_path(original_image_url, image_root)
        
        # extract text and choices
        suggestion_list = []
        choices_list = []
        for annotation in task.get('annotations', []):
            for result_item in annotation.get('result', []):
                if result_item['from_name'] == 'notes':
                    text = result_item['value'].get('text', [])
                    suggestion_list.extend(text)
                elif result_item['from_name'] == 'dynamic_reasons':
                    choices = result_item['value'].get('choices', [])
                    choices_list.extend(choices)
        
        results.append({
            "revisions": suggestion_list,
            "choices": choices_list,
            "image_path": local_image_path
        })
    
    return results



def get_annotation_path(image_path, output_folder):
    filename = os.path.basename(image_path)
    output_filename = os.path.splitext(filename)[0] + ".txt"
    subfolder = os.path.basename(os.path.dirname(image_path))

    current_folder = os.path.join(output_folder, subfolder)
    if not os.path.exists(current_folder):
        os.makedirs(current_folder, exist_ok=True)

    output_path = os.path.join(current_folder, output_filename)

    return output_path

def revise_annotation(metadata, output_folder):

    image_path = metadata["image_path"]
    original_text = '\n'.join([f"<begin_of_point>\n{choice}<end_of_point>" for choice in metadata["choices"]])
    manual_revisions = '\n'.join([f"{i + 1}. {revision}" for i, revision in enumerate(metadata["revisions"])])

    # save to new image path
    import shutil
    image_name = os.path.basename(image_path)
    subfolder = os.path.basename(os.path.dirname(image_path))
    image_subfolder = os.path.join("mygenImages_benchmark_selected_20250403", subfolder) 
    new_image_path = os.path.join(image_subfolder, image_name)
    os.makedirs(image_subfolder, exist_ok=True)
    shutil.copy(image_path, new_image_path)

    prompt = manual_revise_prompt.format(original_text=original_text, manual_revisions=manual_revisions)
    print(prompt)
    print("="*70)

    output_path = get_annotation_path(image_path, output_folder)
    if os.path.exists(output_path):
        return

    refined_text = gpt4o_response(prompt, image_path)

    if refined_text:
        with open(output_path, "w") as f:
            f.write(refined_text)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process images with GPT-4 and a prompt template (parallel processing).")
    parser.add_argument("--json_path", default="label_studio_manual.json", help="Labeled JSON data from label-studio")
    parser.add_argument("--image_root", default="generated_images", help="Path to the folder containing images.")
    parser.add_argument("--annotation_root", default="generated_annotation_high_level_revised_manual", help="Path to the folder where high-level output text files will be saved.")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of worker threads. (Default: number of CPU cores)")
    args = parser.parse_args()

    os.makedirs(args.annotation_root, exist_ok=True)

    data = extract_data(args.json_path, args.image_root)

    # Use a thread pool to process images in parallel
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        list(tqdm(executor.map(revise_annotation, data, [args.annotation_root] * len(data)),
                  total=len(data), desc="Processing images"))