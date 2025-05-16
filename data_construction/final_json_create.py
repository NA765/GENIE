import os
import json
import random
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset for image authenticity classification")
    parser.add_argument("--fake_image_root", type=str, default="generated_images",
                        help="Root directory of fake images")
    parser.add_argument("--fake_annotation_root", type=str, default="generated_annotation_final",
                        help="Root directory of fake image annotations")
    parser.add_argument("--real_image_root", type=str, default="real_images",
                        help="Root directory of real images")
    parser.add_argument("--real_annotation_root", type=str, default="real_images_final",
                        help="Root directory of real image annotations")
    parser.add_argument("--max_num", type=int, default=None,
                        help="Maximum number of samples per class (default: None)")
    parser.add_argument("--split", action="store_true",
                        help="Whether to split into train and test sets (default: False)")
    parser.add_argument("--output_train_json", type=str, default="final_train_data.json",
                        help="Output path for training JSON file")
    parser.add_argument("--output_test_json", type=str, default="final_test_data.json",
                        help="Output path for test JSON file")
    parser.add_argument("--output_combined_json", type=str,
                        default="final_train_data.json",
                        help="Output path for combined JSON file WHEN NOT SPLITING")
    return parser.parse_args()


questions = [
    "Is this image authentic or artificially generated? Please analyze the low-level features such as texture sharpness, lighting consistency, and noise patterns. Additionally, evaluate any high-level anomalies like unnatural object placement or perspective distortions. Based on your findings, provide a conclusion about its authenticity.",
    "Can you determine if this picture is real or digitally manipulated? Start by examining low-level characteristics like color gradients, edge smoothness, and shadow alignment. Then, assess high-level issues such as illogical scene composition or mismatched elements. Finally, state whether you believe the image is genuine or synthetic.",
    "Does this image depict a real scenario or is it artificially created? Investigate low-level indicators like pixelation artifacts, reflection accuracy, and chromatic aberrations. Also, scrutinize high-level problems like contextual mismatches or implausible proportions. Conclude by stating whether the image appears to be real or synthetic.",
    "Could you verify whether this photo is authentic or computer-generated? Analyze low-level aspects such as JPEG compression artifacts, anti-aliasing quality, and specular highlights. Next, identify high-level discrepancies like inconsistent scale relationships or impossible physics. Provide a final verdict on whether the image is real or fake.",
    "Is this visual content captured in reality or synthesized using AI tools? Focus first on low-level cues such as fine-grained textures, depth-of-field effects, and lens flares. Then, look for high-level mistakes like conflicting environmental details or human poses that defy anatomy. Afterward, determine whether the image is likely real or artificial.",
    "How can we tell if this image represents a real-world situation or was constructed synthetically? Examine low-level traits like blurring consistency, micro-details in surfaces, and tonal balance. Simultaneously, check for high-level flaws like contradictory lighting sources or objects violating spatial logic. Summarize your assessment of the image's origin.",
    "Is there evidence suggesting that this image is either real or fabricated? Begin with low-level inspections of resolution uniformity, aliasing effects, and gradient transitions. Move on to detecting high-level oddities like misplaced shadows or objects behaving unnaturally. End by justifying your decision regarding the image’s legitimacy.",
    "What leads you to conclude that this image is genuine or artificially produced? First, consider low-level attributes like contrast levels, artifact distribution, and repetitive patterns. Then, focus on high-level concerns like improbable interactions between subjects or background anomalies. Share your reasoning and declare whether the image is real or synthetic.",
    "Based on technical analysis, is this image a true representation of reality or a digital creation? Study low-level factors like grain structure, blur coherence, and specular reflections. For high-level evaluation, highlight any narrative inconsistencies or geometric irregularities. Deliver a conclusive statement about the nature of the image.",
    "Would you classify this image as real or synthetic? Scrutinize low-level properties such as luminance variation, interpolation artifacts, and material fidelity. Additionally, detect high-level red flags like temporal mismatches or unrealistic behaviors within the frame. Use these observations to decide and explain your stance."
]


def process_images_and_annotations(image_root, annotation_root, label, data):
    image_name_roots = os.listdir(image_root)
    for image_name_root in tqdm(image_name_roots, desc=f"Processing {'real' if label == 0 else 'fake'} images"):
        if label == 1:
            cur_image_root = os.path.join(image_root, image_name_root)
            for image_name in os.listdir(cur_image_root):
                image_path = os.path.join(cur_image_root, image_name)
                annotation_name = os.path.splitext(image_name)[0] + ".txt"
                annotation_path = os.path.join(annotation_root, image_name_root, annotation_name)

                if not os.path.exists(annotation_path):
                    print(f"Annotation file missing for {image_name}")
                    continue

                with open(annotation_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()

                entry = {
                    "id": os.path.splitext(image_name)[0],
                    "image": image_path,
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{random.choice(questions)}"},
                        {"from": "gpt", "value": text}
                    ],
                    "label": label
                }

                data.append(entry)
        elif label == 0:
            image_path = os.path.join(image_root, image_name_root)
            annotation_name = os.path.splitext(image_name_root)[0] + ".txt"
            annotation_path = os.path.join(annotation_root, annotation_name)

            if not os.path.exists(annotation_path):
                print(f"Annotation file missing for {image_name_root}")
                continue

            with open(annotation_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            if text.endswith("<to_be_filtered>"):
                print(f"Filtered real image: {image_name_root} due to <to_be_filtered>")
                continue

            entry = {
                "id": os.path.splitext(image_name_root)[0],
                "image": image_path,
                "conversations": [
                    {"from": "human", "value": f"<image>\n{random.choice(questions)}"},
                    {"from": "gpt", "value": text}
                ],
                "label": label
            }

            data.append(entry)


def main(args):
    data = []
    process_images_and_annotations(args.fake_image_root, args.fake_annotation_root, label=1, data=data)
    process_images_and_annotations(args.real_image_root, args.real_annotation_root, label=0, data=data)

    real_data = [item for item in data if item["label"] == 0]
    fake_data = [item for item in data if item["label"] == 1]

    random.shuffle(real_data)
    if args.max_num:
        real_data = real_data[:args.max_num]

    random.shuffle(fake_data)
    if args.max_num:
        fake_data = fake_data[:args.max_num]

    combined_data = real_data + fake_data
    random.shuffle(combined_data)

    if args.split:
        split_index = int(len(combined_data) * 0.8)
        train_data, test_data = combined_data[:split_index], combined_data[split_index:]

        with open(args.output_train_json, "w", encoding="utf-8") as f:
            json.dump(train_data, f, indent=4, ensure_ascii=False)

        with open(args.output_test_json, "w", encoding="utf-8") as f:
            json.dump(test_data, f, indent=4, ensure_ascii=False)

        print(f"Training data has been saved to {args.output_train_json}")
        print(f"Testing data has been saved to {args.output_test_json}")

    else:
        with open(args.output_combined_json, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, indent=4, ensure_ascii=False)

        print(f"Combined data has been saved to {args.output_combined_json}")

    # Print statistics
    total = len(combined_data)
    real_count = len(real_data)
    fake_count = len(fake_data)
    print(f"\nDataset Statistics:")
    print(f"- Total images: {total}")
    print(f"- Real images: {real_count} ({real_count / total:.2%})")
    print(f"- Fake images: {fake_count} ({fake_count / total:.2%})")


if __name__ == "__main__":
    args = parse_args()
    main(args)