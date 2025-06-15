import json
import difflib # For Longest Common Subsequence (LCS) based similarity
from collections import defaultdict
import matplotlib.pyplot as plt
import os
# from sklearn.metrics import f1_score, precision_score, recall_score # Not used directly in this version's F1

# Import config if needed for standalone use, but primarily used by train.py now
# import config

def parse_quadruplet_string(quad_string):
    """Parses a single quadruplet string into its components."""
    try:
        # Normalize spaces around [END] before stripping
        quad_string = quad_string.replace(" [END]", "[END]").replace("[END] ", "[END]")
        if quad_string.endswith("[END]"):
            quad_string = quad_string[:-len("[END]")].strip()
        else: # Handle cases where [END] might be missing or malformed
            # A more robust cleanup might be needed depending on typical model errors
            if quad_string.strip().endswith("]"):
                quad_string = quad_string.strip()[:-1]
                if quad_string.strip().endswith("[END"): quad_string = quad_string.strip()[:-4]
                elif quad_string.strip().endswith("END"): quad_string = quad_string.strip()[:-3]

        parts = [p.strip() for p in quad_string.split(" | ")]
        if len(parts) == 4:
            return {
                "target": parts[0],
                "argument": parts[1],
                "group": parts[2],
                "hateful": parts[3],
            }
    except Exception: # pylint: disable=broad-except
        # print(f"Warning: Could not parse quadruplet string: '{quad_string_original}'")
        pass
    return None


def parse_output_line(line):
    """Parses a line that might contain multiple quadruplets separated by [SEP]."""
    # Normalize spaces around [SEP]
    line = line.replace(" [SEP] ", "[SEP]")
    quad_strings = line.strip().split("[SEP]")
    parsed_quads = []
    for q_str_raw in quad_strings:
        q_str = q_str_raw.strip()
        if not q_str: continue
        parsed = parse_quadruplet_string(q_str)
        if parsed:
            parsed_quads.append(parsed)
    return parsed_quads


def get_string_similarity(s1, s2):
    """Calculates string similarity based on LCS ratio."""
    if not s1 and not s2: return 1.0
    if not s1 or not s2: return 0.0
    return difflib.SequenceMatcher(None, s1, s2).ratio()

def calculate_f1_metrics_from_lists(pred_quad_lists, gt_quad_lists):
    """
    Calculates F1 metrics given lists of predicted quadruplets and ground truth quadruplets.
    Each item in pred_quad_lists and gt_quad_lists corresponds to a document,
    and contains a list of parsed quadruplet dicts for that document.
    """
    tp_hard, fp_hard, fn_hard = 0, 0, 0
    tp_soft, fp_soft, fn_soft = 0, 0, 0

    for doc_idx in range(len(gt_quad_lists)):
        pred_quads_doc = pred_quad_lists[doc_idx] if doc_idx < len(pred_quad_lists) else []
        gt_quads_doc = gt_quad_lists[doc_idx]

        # These sets track which GT quads in the current doc have been matched
        matched_gt_indices_hard_doc = set()
        matched_gt_indices_soft_doc = set()

        # Calculate FPs for the current document initially
        current_doc_fp_hard = len(pred_quads_doc)
        current_doc_fp_soft = len(pred_quads_doc)

        for p_quad in pred_quads_doc:
            hard_match_found_for_this_p_quad = False
            soft_match_found_for_this_p_quad = False

            # Try to find a hard match
            for gt_idx, gt_quad in enumerate(gt_quads_doc):
                if gt_idx not in matched_gt_indices_hard_doc:
                    if (p_quad["target"] == gt_quad["target"] and
                        p_quad["argument"] == gt_quad["argument"] and
                        p_quad["group"] == gt_quad["group"] and
                        p_quad["hateful"] == gt_quad["hateful"]):
                        tp_hard += 1
                        matched_gt_indices_hard_doc.add(gt_idx)
                        hard_match_found_for_this_p_quad = True
                        break # Each predicted quad can match at most one GT quad

            # Try to find a soft match (independent of hard match for accounting)
            for gt_idx, gt_quad in enumerate(gt_quads_doc):
                if gt_idx not in matched_gt_indices_soft_doc: # GT quad not already soft-matched
                    if (p_quad["group"] == gt_quad["group"] and
                        p_quad["hateful"] == gt_quad["hateful"] and
                        get_string_similarity(p_quad["target"], gt_quad["target"]) > 0.5 and
                        get_string_similarity(p_quad["argument"], gt_quad["argument"]) > 0.5):
                        tp_soft += 1
                        matched_gt_indices_soft_doc.add(gt_idx)
                        soft_match_found_for_this_p_quad = True
                        break # Each predicted quad can match at most one GT quad for soft match

            if hard_match_found_for_this_p_quad:
                current_doc_fp_hard -= 1
            if soft_match_found_for_this_p_quad:
                current_doc_fp_soft -= 1

        fp_hard += current_doc_fp_hard
        fp_soft += current_doc_fp_soft

        fn_hard += len(gt_quads_doc) - len(matched_gt_indices_hard_doc)
        fn_soft += len(gt_quads_doc) - len(matched_gt_indices_soft_doc)

    precision_hard = tp_hard / (tp_hard + fp_hard) if (tp_hard + fp_hard) > 0 else 0
    recall_hard = tp_hard / (tp_hard + fn_hard) if (tp_hard + fn_hard) > 0 else 0
    f1_hard = 2 * (precision_hard * recall_hard) / (precision_hard + recall_hard) if (precision_hard + recall_hard) > 0 else 0

    precision_soft = tp_soft / (tp_soft + fp_soft) if (tp_soft + fp_soft) > 0 else 0
    recall_soft = tp_soft / (tp_soft + fn_soft) if (tp_soft + fn_soft) > 0 else 0
    f1_soft = 2 * (precision_soft * recall_soft) / (precision_soft + recall_soft) if (precision_soft + recall_soft) > 0 else 0
    
    overall_score = (f1_hard + f1_soft) / 2

    results = {
        "hard_match": {"precision": precision_hard, "recall": recall_hard, "f1": f1_hard, "tp": tp_hard, "fp": fp_hard, "fn": fn_hard},
        "soft_match": {"precision": precision_soft, "recall": recall_soft, "f1": f1_soft, "tp": tp_soft, "fp": fp_soft, "fn": fn_soft},
        "overall_score": overall_score
    }
    return results


def plot_evaluation_scores(results, output_dir, plot_filename="validation_f1_scores.png"):
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = ['Precision', 'Recall', 'F1-Score']
    hard_scores = [results['hard_match']['precision'], results['hard_match']['recall'], results['hard_match']['f1']]
    soft_scores = [results['soft_match']['precision'], results['soft_match']['recall'], results['soft_match']['f1']]

    x_labels = metrics
    x_pos = range(len(x_labels)) # Use range for x positions
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 7)) # Increased height for title
    rects1 = ax.bar([i - width/2 for i in x_pos], hard_scores, width, label='Hard Match')
    rects2 = ax.bar([i + width/2 for i in x_pos], soft_scores, width, label='Soft Match')

    ax.set_ylabel('Scores')
    ax.set_title(f'Validation Set Evaluation Metrics\nOverall Score (Avg F1): {results["overall_score"]:.4f}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.set_ylim(0, max(1.05, max(hard_scores + soft_scores) * 1.1)) # Adjust y-lim dynamically

    def autolabel(rects_list):
        for r in rects_list:
            height = r.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(r.get_x() + r.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make space for title
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path)
    print(f"Validation evaluation F1 scores plot saved to {plot_path}")
    plt.close()

if __name__ == "__main__":
    print("evaluate.py now primarily serves as a utility module.")
    print("Its functions (calculate_f1_metrics_from_lists, plot_evaluation_scores, etc.)")
    print("are imported and used by train.py for evaluating the validation set.")
    print("To run a standalone evaluation, you would need prediction and ground truth files,")
    print("and adapt the main block here or call the functions appropriately.")

    # Example of how it could be used standalone if you had files:
    #
    # from config import PREDICTIONS_FILE, TEST_GROUND_TRUTH_FILE, EVALUATION_PLOTS_DIR
    #
    # if not os.path.exists(PREDICTIONS_FILE) or not os.path.exists(TEST_GROUND_TRUTH_FILE):
    #     print(f"Error: Predictions file or ground truth file not found.")
    # else:
    #     print(f"Evaluating predictions from: {PREDICTIONS_FILE}")
    #     print(f"Against ground truth: {TEST_GROUND_TRUTH_FILE}")
    #
    #     with open(PREDICTIONS_FILE, 'r', encoding='utf-8') as f_pred:
    #         pred_lines = [line.strip() for line in f_pred.readlines()]
    #     parsed_pred_quads_lists = [parse_output_line(line) for line in pred_lines]
    #
    #     gt_data_list = [] # list of {"id": ..., "output": ...}
    #     with open(TEST_GROUND_TRUTH_FILE, 'r', encoding='utf-8') as f_gt:
    #         gt_json_data = json.load(f_gt)
    #         # Assuming gt_json_data is a list of dicts with "output" field
    #         gt_data_list = [parse_output_line(item["output"]) for item in gt_json_data]
    #
    #     results = calculate_f1_metrics_from_lists(parsed_pred_quads_lists, gt_data_list)
    #
    #     if results:
    #         print("\nEvaluation Results:")
    #         # ... (print results as before) ...
    #         plot_evaluation_scores(results, EVALUATION_PLOTS_DIR, "standalone_eval_scores.png")
    #     else:
    #         print("Evaluation could not be completed.")