import numpy as np
import json
from tqdm import tqdm
from argparse import ArgumentParser

import evaluate
from sentence_transformers import SentenceTransformer, util

from utils.gpt4o import gpt4o_response
from utils.utils import *


def parse_args():
    parser = ArgumentParser(description="Compute scores of evaluation metrics")
    parser.add_argument("--annotation_file", default="benchmark_test_data_result.json", help="Path to the json file containing ground truth and generated text")
    parser.add_argument("--metrics", default="sentence_transformer", help="Path to the folder where output text files will be saved.",
                        choices=["sentence_transformer, bleu@1, bleu@2, bleu@3, bleu@4, rouge, meteor, gpt_4o"])
    args = parser.parse_args()

    return args



def get_pairwise_score(point_gt, point_gen):

    pairwise_prompt = """Analyze and compare the semantic similarity between the two sentences provided below. Evaluate their meaning, context, and structure to determine how closely they match. Return a similarity score as a value between 0 and 1, where 0 means no similarity and 1 means identical in meaning. Put your similarity score within the `\\boxed{{}}`.

        Sentence 1: "{sentence_1}"
        Sentence 2: "{sentence_2}"
        """
    prompt = pairwise_prompt.format(sentence_1=point_gt, sentence_2=point_gen)
    response = gpt4o_response(prompt)
    
    if not response:
        return 0.0
    
    boxed_contents = get_boxed_content(response)
    if not boxed_contents:
        return 0.0

    try:
        similarity_score = float(boxed_contents[0])
        return similarity_score
    except ValueError:
        return 0.0
    

def compute_score_matrix(list1, list2, metric_name="bleu", bleu_order=None):
    scorer = evaluate.load(metric_name)
    M, N = len(list1), len(list2)
    score_matrix = np.zeros((M, N))

    for i in tqdm(range(M), desc=f"Computing {metric_name} scores"):
        for j in range(N):
            ref = [[list1[i]]] if metric_name == "bleu" else [list1[i]]
            pred = list2[j]
            if metric_name == "bleu":
                if bleu_order is None:
                    bleu_order = 4
                score = scorer.compute(predictions=[pred], references=ref, max_order=bleu_order)
            else:
                score = scorer.compute(predictions=[pred], references=ref)
                if metric_name == "rouge":
                    score[metric_name] = score["rougeL"]
            score_matrix[i, j] = score[metric_name]
    return score_matrix




def compute_metrics(label_gt, label_gen, points_gt, points_gen, metrics, threshold=0.7):

    assert label_gt.lower() in ["ai-generated", "real"] and label_gen.lower() in ["ai-generated", "real"]

    if len(points_gt) == 0:
        return None

    M, N = len(points_gt), len(points_gen)

    accuracy = (label_gt.lower() == label_gen.lower())

    if not accuracy:
        return 0.0, 0.0, 0.0, 1.0
    elif accuracy and label_gt.lower() == "real":
        return 1.0, 1.0, 1.0, 0.0

    accuracy = 1.0

    # compute pairwise score from GPT-4o (slow)
    if metrics == "gpt_4o":
        score_matrix = np.zeros((M, N))
        for i in range(M):
            for j in range(N):
                score_matrix[i, j] = get_pairwise_score(points_gt[i], points_gen[j])
    

    # Sentence Transformers
    elif metrics == "sentence_transformers":
        embedding1 = model.encode(points_gt, convert_to_tensor=True)
        embedding2 = model.encode(points_gen, convert_to_tensor=True)
        score_matrix = util.cos_sim(embedding1, embedding2).cpu().numpy()

    # NLP metrics (BLEU, ROUGE, METEOR)
    elif metrics in ["bleu@1", "bleu@2", "bleu@3", "bleu@4", "rouge", "meteor"]:
        if metrics.startswith("bleu"):   
            bleu_order = int(metrics.split('@')[1]) if metrics.startswith("bleu") else None
            score_matrix = compute_score_matrix(points_gt, points_gen, metric_name="bleu", bleu_order=bleu_order)
        else:
            score_matrix = compute_score_matrix(points_gt, points_gen, metric_name=metrics)


    # compute score list
    score_list = np.zeros(M)
    k = 0
    while k < min(M, N):
        i_max, j_max = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
        score_list[i_max] = score_matrix[i_max, j_max]
        score_matrix[i_max, :] = 0
        score_matrix[:, j_max] = 0
        k += 1

    # compute match score and richness score
    match_score = np.mean(score_list)
    richness_score = np.sum(score_list >= threshold) / M

    # compute halluciation rate
    halluciation_rate = 1 - np.sum(score_list >= threshold) / N if N > 0 else 0.0


    return accuracy, match_score, richness_score, halluciation_rate



if __name__ == "__main__":

    args = parse_args()

    with open(args.annotation_file, 'r') as f:
        annotations = json.load(f)

    model = SentenceTransformer('all-MiniLM-L6-v2')

    total_accuracy, total_match_score, total_richness_score, total_halluciation_rate = [], [], [], []

    for annotation in tqdm(annotations):
        image_path = annotation["image_path"]
        ground_truth = annotation["ground_truth"]
        generated = annotation["generated"]
        label = annotation["label"]

        # extract high-level errors
        ground_truth_high_level = extract_content_by_regex(ground_truth, start_marker="<begin_of_high_level_errors>", end_marker="<end_of_high_level_errors>")
        generated_high_level = extract_content_by_regex(generated, start_marker="<begin_of_high_level_errors>", end_marker="<end_of_high_level_errors>")

        ground_truth_high_level = ground_truth_high_level if ground_truth_high_level else ground_truth
        generated_high_level = generated_high_level if generated_high_level else generated

        _, points_gt, _ = parse_text(ground_truth_high_level)
        _, points_gen, _ = parse_text(generated_high_level)

        label_gt = get_boxed_content(ground_truth)
        label_gen = get_boxed_content(generated)

        label_gt = label_gt if label_gt else "real"
        label_gen = label_gen if label_gen else "real"

        results = compute_metrics(label_gt, label_gen, points_gt, points_gen, args.metrics)
        if results:
            accuracy, match_score, richness_score, halluciation_rate = results
            total_match_score.append(match_score)
            total_richness_score.append(richness_score)
            total_halluciation_rate.append(halluciation_rate)
            total_accuracy.append(accuracy)


    avg_match_score = sum(total_match_score) / len(total_match_score)
    avg_richness_score = sum(total_richness_score) / len(total_richness_score)
    avg_halluciation_rate = sum(total_halluciation_rate) / len(total_halluciation_rate)
    avg_accuracy = sum(total_accuracy) / len(total_accuracy)

    print("avg_match_score:", avg_match_score)
    print("avg_richness_score:", avg_richness_score)
    print("avg_halluciation_rate:", avg_halluciation_rate)
    print("avg_accuracy:", avg_accuracy)

    with open("tmp_eval_result.txt", "w") as f:
        f.write(f"avg_match_score:{avg_match_score}\n")
        f.write(f"avg_richness_score:{avg_richness_score}\n")
        f.write(f"avg_halluciation_rate:{avg_halluciation_rate}\n")
        f.write(f"avg_accuracy:{avg_accuracy}\n")



