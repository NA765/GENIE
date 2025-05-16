import os
import json
import re
import argparse

from utils.constants import *



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


if __name__ =='__main__':

    parser = argparse.ArgumentParser(description="Process images with GPT-4 and a prompt template (parallel processing).")
    parser.add_argument("--image_dir", default="label_studio_manual.json", help="Path to the folder containing images.")
    parser.add_argument("--text_dir", default="generated_annotation_high_level_revised_manual", help="Path to the folder where high-level text files are stored.")
    parser.add_argument("--output_path", default="labelstudio_tasks.json", help="Path to the output JSON file for label-studio labeling.")
    args = parser.parse_args()


    image_dir = args.image_dir
    text_dir = args.text_dir
    output_path = args.output_path

    tasks = []

    for subfolder in sorted(os.listdir(image_dir)):
        image_subdir = os.path.join(image_dir, subfolder)
        text_subdir = os.path.join(text_dir, subfolder)
        if not os.path.isdir(image_subdir) or not os.path.isdir(text_subdir):
            continue
        for filename in sorted(os.listdir(image_subdir)):
            if filename.endswith(".png"):
                task_id = filename.replace(".png", "")
                image_url = f"http://{SERVER_IP}:{PORT}/images/{subfolder}/{filename}"
                text_filename = filename.replace(".png", ".txt")
                text_filepath = os.path.join(text_subdir, text_filename)
                if os.path.exists(text_filepath):
                    with open(text_filepath, "r", encoding="utf-8") as f:
                        text_content = f.read().strip()
                else:
                    text_content = ""
                prefix, point_list, conclusion = parse_text(text_content)
                
                task_data = {"image": image_url, "task_id": task_id}
                if prefix:
                    task_data["prefix"] = prefix
                task_data["points"] = list()
                for idx, pt in enumerate(point_list):
                    task_data["points"].append({"value": pt})
                if conclusion:
                    task_data["conclusion"] = conclusion
                else:
                    task_data["conclusion"] = "No conclusion"
                tasks.append({"data": task_data})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=4, ensure_ascii=False)

    print(f"{output_path} generated, {len(tasks)} tasks in total.")
