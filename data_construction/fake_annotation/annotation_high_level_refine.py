import os
import re
import argparse
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from utils.gpt4o import gpt4o_response


refine_prompt = """You have been given an annotated text of a synthesized image. The text follows a structured format, which consists of:  

- **Prefix** (which might be empty)  
- **Points** (each enclosed within `<begin_of_point>` and `<end_of_point>`)  
- **Conclusion** (which might be empty)  

Your task is to analyze each **point** individually and assess its validity based on the provided image. Specifically, you should:  

1. **Determine whether the described elements in each point are clearly present in the image.**  
   - If a point states that this error is not present in the image, the point should be marked for removal, otherwise do not arbitrarily remove a point.
2. **Assess whether the point’s description is factually accurate and sufficiently precise.**  
   - If a point contains details lacking clarity, suggest improvements.  

**Output Format:**  
- Provide a detailed analysis of each point, identifying any redundancy or unclarity.  
- At the end of your response, format the necessary modifications using JSON, enclosed within `<begin_of_json>` and `<end_of_json>`.  
- The JSON format should be directly usable without additional formatting or annotations.  

**JSON Format Example:**  
```  
<begin_of_json>  
[  
    {  
        "type": "remove",  
        "location": 1  
    },  
    {  
        "type": "revise",  
        "location": 3,  
        "content": "<revised_content>"  
    },
]  
<end_of_json>  
```
- `"type"` must be either `"remove"` (to delete a point), `"revise"` (to modify a point).  
- `"location"` refers to the point’s position, starting from 1.  
- If `"type"` is `"revise"`, `"content"` must contain the corrected version of the point.
- Ensure that after revising or removing points, the sequence of points remains in order.
"""


def parse_text(text):
    text = text.strip()
    
    sections = re.split(r"<begin_of_point>", text)
    prefix = sections[0].strip() if sections[0] else None
    
    points = []
    conclusion = None
    
    for section in sections[1:]:
        parts = section.split("<end_of_point>", 1)
        if len(parts) == 2:
            points.append(parts[0].strip())
            conclusion = parts[1].strip() if parts[1] else None
        else:
            points.append(parts[0].strip())
    
    return prefix, points, conclusion


def join_text(prefix, points, conclusion):
    sections = []
    
    if prefix:
        sections.append(prefix.strip())  # Add prefix if it exists

    for point in points:
        if point:
            sections.append(f"<begin_of_point>\n{point.strip()}\n<end_of_point>")  # Format each point
    
    if conclusion:
        sections.append(conclusion.strip())  # Add conclusion if it exists
    
    return "\n\n".join(sections)  # Join sections with two newlines


def get_refine_prompt(text):
    formated_refine_prompt = refine_prompt + f"Annotated text:\n{text}"
    return formated_refine_prompt


def get_suggestions(text):
    start_marker = "<begin_of_json>"
    end_marker = "<end_of_json>"

    start_index = text.find(start_marker)
    end_index = text.find(end_marker)

    if start_index == -1 or end_index == -1 or start_index >= end_index:
        return None

    json_string = text[start_index + len(start_marker):end_index].strip()

    try:
        json_object = json.loads(json_string)
        return json_object
    except json.JSONDecodeError as e:
        return None


def refine_text(image_path, text):
    formated_refine_prompt = get_refine_prompt(text)
    refined_response = gpt4o_response(formated_refine_prompt, image_path)
    suggestions = get_suggestions(refined_response)

    if suggestions is None:
        return text
    
    prefix, points, conclusion = parse_text(text)
    try:
        for suggestion in suggestions:
            location = suggestion["location"] - 1  # index starts from 0
            suggestion_type = suggestion["type"]
            if suggestion_type == "revise":
                points[location] = suggestion["content"]
            elif suggestion_type == "remove":
                points[location] = None
            else:
                print("Invalid type:", suggestion_type)
    except Exception as e:
        print("Error:", e)

    refined_text = join_text(prefix, [p for p in points if p is not None], conclusion)
    return refined_text


def process_single_image(image_path, annotation_path, new_annotation_path):
    """
    单个图像和标注文件的处理函数。
    """
    if not os.path.exists(annotation_path):
        print(f"Annotation file missing for {image_path}")
        return
    
    if os.path.exists(new_annotation_path):
        print(f"Annotation file already exists: {new_annotation_path}")
        return

    with open(annotation_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    refined_text = refine_text(image_path, text)

    with open(new_annotation_path, "w", encoding="utf-8") as f:
        f.write(refined_text)


def process_fake_annotations(image_root, annotation_root, new_annotation_root, max_workers=None):

    subfolders = [f.name for f in os.scandir(image_root) if f.is_dir()]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for subfolder in tqdm(subfolders, desc="Processing subfolders"):
            image_subfolder = os.path.join(image_root, subfolder)
            annotation_subfolder = os.path.join(annotation_root, subfolder)
            new_annotation_subfolder = os.path.join(new_annotation_root, subfolder)

            os.makedirs(new_annotation_subfolder, exist_ok=True)

            for image_name in os.listdir(image_subfolder):
                image_path = os.path.join(image_subfolder, image_name)
                annotation_name = os.path.splitext(image_name)[0] + ".txt"
                annotation_path = os.path.join(annotation_subfolder, annotation_name)
                new_annotation_path = os.path.join(new_annotation_subfolder, annotation_name)

                futures.append(executor.submit(
                    process_single_image, image_path, annotation_path, new_annotation_path
                ))

        for future in tqdm(futures, desc="Processing images"):
            future.result()


def main():
    parser = argparse.ArgumentParser(description="Process annotations for fake images.")
    parser.add_argument("--image_root", type=str, default="generated_images",
                        help="Root directory containing fake image subfolders.")
    parser.add_argument("--annotation_root", type=str, default="generated_annotation_high_level_norefined",
                        help="Root directory containing original annotation subfolders.")
    parser.add_argument("--output_root", type=str, default="generated_annotation_high_level_refined",
                        help="Root directory to save processed annotations.")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="Maximum number of worker threads for parallel processing.")

    args = parser.parse_args()

    print(f"Image root: {args.image_root}")
    print(f"Annotation root: {args.annotation_root}")
    print(f"Output root: {args.output_root}")
    print(f"Max workers: {args.max_workers}")

    process_fake_annotations(args.image_root, args.annotation_root, args.output_root, args.max_workers)


if __name__ == "__main__":
    main()
