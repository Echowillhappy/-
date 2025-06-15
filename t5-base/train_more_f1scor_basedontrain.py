# train_more_f1scor_basedontrain.py

import os
import json
import argparse
import logging
from pathlib import Path
import sys # 导入 sys 模块
from unittest.mock import patch # 用于在 Jupyter 环境中模拟 argparse

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from difflib import SequenceMatcher # 新增：用于F1分数计算
import numpy as np # 新增：用于F1分数计算

# 设置日志 (在 Jupyter 中可能会有重复日志，但为了独立运行保留)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义有效的目标群体 (沿用您最新提供的文件中的定义)
VALID_TARGETED_GROUPS = ['Region', 'Racism', 'Sexism', 'LGBTQ', 'others', 'non-hate']
TARGETED_GROUP_MAPPING = {
    "地域": "Region",
    "种族": "Racism",
    "性别": "Sexism",
    "LGBTQ": "LGBTQ",
    "其他": "others",
    "non-hate": "non-hate"
}

def format_quadruplet_to_string(quad):
    """
    将单个四元组字典格式化为所需的字符串格式。
    确保 'targeted_group' 是有效类别之一。
    在每个四元组的末尾添加 [END]。
    """
    target = quad.get("target", "NULL")
    argument = quad.get("argument", "")
    targeted_group_raw = quad.get("targeted_group", "non-hate")
    hateful = quad.get("hateful", "non-hate")

    targeted_group = TARGETED_GROUP_MAPPING.get(targeted_group_raw, targeted_group_raw)
    if targeted_group not in VALID_TARGETED_GROUPS:
        logger.warning(f"检测到无效的 targeted_group '{targeted_group_raw}'。将其映射为 'others'。原始四元组: {quad}")
        targeted_group = 'others'
    
    # 优化：确保 target 和 argument 如果为空字符串，也显示为 "NULL"
    target_str = target if target and str(target).strip() else "NULL"
    argument_str = argument if argument and str(argument).strip() else "NULL"
    
    return f"{target_str} | {argument_str} | {targeted_group} | {hateful} [END]"

def create_target_string(quadruplets_list):
    """
    从四元组字典列表中创建完整的输出目标字符串。
    使用 [SEP] 连接，不添加额外的 [END]。
    """
    if not quadruplets_list:
        logger.warning("发现空四元组列表。生成默认的 non-hate 字符串。")
        return "NULL | NULL | non-hate | non-hate [END]"

    formatted_quads = [format_quadruplet_to_string(quad) for quad in quadruplets_list]
    return " [SEP] ".join(formatted_quads)


class HateSpeechQuadrupletDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_input_length, max_target_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label_str = self.labels[idx]

        input_encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_input_length,
            return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            label_str,
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt"
        )

        labels = target_encoding.input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100 # Important for loss calculation

        return {
            "input_ids": input_encoding.input_ids.squeeze(),
            "attention_mask": input_encoding.attention_mask.squeeze(),
            "labels": labels.squeeze()
        }

def load_and_preprocess_data(data_path, tokenizer, max_input_length, max_target_length, val_size=0.1):
    logger.info(f"正在从 {data_path} 加载数据...")
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except Exception as e:
        logger.error(f"加载或解析 {data_path} 时出错: {e}")
        raise

    texts = []
    target_strings = [] # 存储从 JSON 直接读取的、已经格式化好的字符串

    for item in raw_data:
        text = item.get("content")
        # 直接使用 output 字段作为目标字符串，因为它已经是格式化好的
        # 这里不会有额外的类型检查，保持与您最新提供的原始文件一致
        formatted_output_string = item.get("output") 

        if text is None or formatted_output_string is None:
            logger.warning(f"由于缺少 'content' 或 'output'，跳过该项: {item}")
            continue
        
        texts.append(text)
        target_strings.append(formatted_output_string) 

    if not texts:
        logger.error("处理后未找到有效数据。请检查数据格式。")
        raise ValueError("未加载到有效数据。")

    logger.info(f"已加载 {len(texts)} 条样本。")
    if texts:
        logger.info(f"示例处理后的文本: {texts[0]}")
        logger.info(f"示例处理后的目标字符串: {target_strings[0]}")

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, target_strings, test_size=val_size, random_state=42
    )

    train_dataset = HateSpeechQuadrupletDataset(train_texts, train_labels, tokenizer, max_input_length, max_target_length)
    val_dataset = HateSpeechQuadrupletDataset(val_texts, val_labels, tokenizer, max_input_length, max_target_length)
    
    # 返回原始的 val_texts 和 val_labels，它们已经是模型目标字符串格式
    return train_dataset, val_dataset, val_texts, val_labels 

# --- START: F1分数计算辅助函数 (适配 [END] 每四元组格式) ---
def parse_quadruplet_string(quad_str):
    """
    解析单个四元组字符串，去除 [END] 并处理 NULL。
    """
    if not quad_str or not quad_str.strip():
        return None
    
    # 移除可能的多余 [END]，因为原始格式是在每个四元组后有 [END]
    if quad_str.endswith(" [END]"):
        quad_str = quad_str[:-len(" [END]")]
    quad_str = quad_str.strip() # 再次清理空格

    parts = quad_str.split(" | ")
    
    # 如果解析失败，返回 None
    if len(parts) != 4:
        # 针对偶尔可能出现的解析错误，记录更详细的日志
        logger.warning(f"无法解析四元组字符串: '{quad_str}' (期望4个部分，得到 {len(parts)} 个)。返回 None。")
        return None
    
    # 将 "NULL" 字符串转换为 Python 的 None 值，便于比较
    target_val = parts[0] if parts[0] != "NULL" else None
    argument_val = parts[1] if parts[1] != "NULL" else None

    # 确保 targeted_group 在 VALID_TARGETED_GROUPS 中
    targeted_group = TARGETED_GROUP_MAPPING.get(parts[2], parts[2])
    if targeted_group not in VALID_TARGETED_GROUPS:
        targeted_group = 'others' # fallback to 'others'
        
    return {
        "target": target_val,
        "argument": argument_val,
        "targeted_group": targeted_group,
        "hateful": parts[3]
    }

def parse_target_string_to_quadruplets(full_target_string):
    """
    解析完整的包含多个四元组的目标字符串，返回四元组字典列表。
    预期格式是 'QUAD1 [END] [SEP] QUAD2 [END]'
    """
    if not full_target_string or not full_target_string.strip():
        return []
    
    # 首先按 [SEP] 分割，每个部分应该是一个包含 [END] 的四元组字符串
    quad_strings = full_target_string.split(" [SEP] ")
    quadruplets = []
    for qs in quad_strings:
        parsed_quad = parse_quadruplet_string(qs.strip())
        if parsed_quad: # 只添加成功解析的四元组
            quadruplets.append(parsed_quad)
    return quadruplets

def calculate_similarity(s1, s2):
    """计算两个字符串之间的相似度 (Jaccard 或 Levenshtein 距离，这里使用 difflib.SequenceMatcher)。"""
    if s1 is None: s1 = ""
    if s2 is None: s2 = ""
    if not s1 and not s2: # 两个都为空，相似度为1
        return 1.0
    if not s1 or not s2: # 一个为空，一个不为空，相似度为0
        return 0.0
    s = SequenceMatcher(None, s1, s2)
    return s.ratio()

def is_hard_match(pred_quad, gold_quad):
    """判断两个四元组是否硬匹配 (所有字段严格相等)。"""
    if pred_quad is None or gold_quad is None: return False
    return (str(pred_quad.get("target")) == str(gold_quad.get("target")) and
            str(pred_quad.get("argument")) == str(gold_quad.get("argument")) and
            pred_quad.get("targeted_group") == gold_quad.get("targeted_group") and
            pred_quad.get("hateful") == gold_quad.get("hateful"))

def is_soft_match(pred_quad, gold_quad, similarity_threshold=0.5):
    """判断两个四元组是否软匹配 (类别字段相等，字符串字段相似度超过阈值)。"""
    if pred_quad is None or gold_quad is None: return False
    
    # 确保 targeted_group 在比较前被规范化
    pred_targeted_group = TARGETED_GROUP_MAPPING.get(pred_quad.get("targeted_group"), pred_quad.get("targeted_group"))
    gold_targeted_group = TARGETED_GROUP_MAPPING.get(gold_quad.get("targeted_group"), gold_quad.get("targeted_group"))
    
    if pred_targeted_group not in VALID_TARGETED_GROUPS:
        pred_targeted_group = 'others'
    if gold_targeted_group not in VALID_TARGETED_GROUPS:
        gold_targeted_group = 'others'
    
    # 首先检查类别字段是否匹配
    if not (pred_targeted_group == gold_targeted_group and
            pred_quad.get("hateful") == gold_quad.get("hateful")):
        return False
    
    # 然后检查 Target 和 Argument 的相似度
    target_sim = calculate_similarity(pred_quad.get("target"), gold_quad.get("target"))
    argument_sim = calculate_similarity(pred_quad.get("argument"), gold_quad.get("argument"))
    
    return target_sim >= similarity_threshold and argument_sim >= similarity_threshold


def calculate_f1_score_for_batch(predictions_list_of_lists, references_list_of_lists, match_type='hard'):
    """
    计算一个批次中预测和真实四元组列表的 F1、Precision 和 Recall。
    """
    total_true_positives = 0
    total_predicted_quads = 0
    total_ground_truth_quads = 0

    for i in range(len(predictions_list_of_lists)):
        pred_list = predictions_list_of_lists[i]
        gold_list = references_list_of_lists[i]

        total_predicted_quads += len(pred_list)
        total_ground_truth_quads += len(gold_list)

        matched_gold_indices = set() # 跟踪当前样本中已匹配的真实四元组索引
        for pred_quad in pred_list:
            is_matched_in_this_sample = False
            for g_idx, gold_quad in enumerate(gold_list):
                if g_idx in matched_gold_indices:
                    continue # 这个真实四元组已经被之前的预测匹配过

                if match_type == 'hard' and is_hard_match(pred_quad, gold_quad):
                    total_true_positives += 1
                    matched_gold_indices.add(g_idx)
                    is_matched_in_this_sample = True
                    break # 找到匹配，移动到下一个预测
                elif match_type == 'soft' and is_soft_match(pred_quad, gold_quad):
                    total_true_positives += 1
                    matched_gold_indices.add(g_idx)
                    is_matched_in_this_sample = True
                    break # 找到匹配，移动到下一个预测
    
    precision = total_true_positives / total_predicted_quads if total_predicted_quads > 0 else 0
    recall = total_true_positives / total_ground_truth_quads if total_ground_truth_quads > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_metrics(eval_preds, tokenizer, val_texts_raw, val_labels_raw):
    """
    为 Trainer 计算评估指标。这将在评估期间被调用。
    """
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0] # 从元组的第一个元素中获取预测结果

    # 解码预测结果
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # 解码标签 (标签中 -100 的部分是 padding)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 解析预测和真实四元组
    predicted_quad_lists = [parse_target_string_to_quadruplets(pred) for pred in decoded_preds]
    gold_quad_lists = [parse_target_string_to_quadruplets(label) for label in decoded_labels]

    # 计算 F1 分数
    hard_f1_results = calculate_f1_score_for_batch(predicted_quad_lists, gold_quad_lists, match_type='hard')
    soft_f1_results = calculate_f1_score_for_batch(predicted_quad_lists, gold_quad_lists, match_type='soft')

    # 记录一些示例预测和标签 (用于调试，可以根据需要调整输出量)
    logger.info("\n--- 批量示例预测 (用于 compute_metrics) ---")
    for i in range(min(2, len(decoded_preds))): # 仅打印前2个样本
        # 确保索引不越界
        original_text = val_texts_raw[i] if i < len(val_texts_raw) else 'N/A'
        gold_label = decoded_labels[i]
        prediction = decoded_preds[i]
        
        logger.info(f"原始文本: {original_text}")
        logger.info(f"真实标签: {gold_label}")
        logger.info(f"预测结果: {prediction}")
        logger.info("-" * 30)

    metrics = {
        "f1_hard": hard_f1_results["f1"],
        "precision_hard": hard_f1_results["precision"],
        "recall_hard": hard_f1_results["recall"],
        "f1_soft": soft_f1_results["f1"],
        "precision_soft": soft_f1_results["precision"],
        "recall_soft": soft_f1_results["recall"],
    }
    logger.info(f"Compute Metrics 结果: {metrics}")
    return metrics
# --- END: F1分数计算辅助函数 ---


# 修改 main 函数以接受参数字典
def main(args_dict=None):
    if args_dict is None: # 如果没有提供 args_dict，则从命令行解析
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_path", type=str, default="train.json", help="train.json 文件路径")
        parser.add_argument("--model_name_or_path", type=str, default="Langboat/mengzi-t5-base", help="预训练模型名称或路径")
        parser.add_argument("--output_dir", type=str, default="./results_quad_extraction_f1_optimized_bs16_eval32", help="保存训练模型的目录") # 更改默认输出目录以反映批次大小
        parser.add_argument("--num_train_epochs", type=int, default=17, help="训练轮数") # 增加 epoch 数量
        parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="训练批次大小") # 批次大小调整
        parser.add_argument("--per_device_eval_batch_size", type=int, default=32, help="评估批次大小") # 批次大小调整
        parser.add_argument("--learning_rate", type=float, default=3e-5, help="学习率") # 降低学习率
        parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
        parser.add_argument("--max_input_length", type=int, default=128, help="最大输入序列长度")
        parser.add_argument("--max_target_length", type=int, default=64, help="生成时的最大目标序列长度")
        parser.add_argument("--val_size", type=float, default=0.1, help="用于验证的训练数据比例")
        parser.add_argument("--save_steps", type=int, default=200, help="每 X 步保存一次检查点。") # 更频繁地保存和评估
        parser.add_argument("--eval_steps", type=int, default=200, help="每 X 步评估一次。") # 更频繁地保存和评估
        parser.add_argument("--logging_steps", type=int, default=50, help="每 X 步记录一次日志。") # 更频繁地日志
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数") # 调整梯度累积步数
        parser.add_argument("--fp16", action="store_true", help="使用 FP16 训练")
        parser.add_argument("--early_stopping_patience", type=int, default=8, help="早停的耐心值") # 增加早停耐心
        args = parser.parse_args()
    else:
        # 将字典转换为 argparse.Namespace 对象
        args = argparse.Namespace(**args_dict)

    logger.info(f"训练参数: {args}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"正在从 {args.model_name_or_path} 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    logger.info(f"正在从 {args.model_name_or_path} 加载模型...")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    
    # load_and_preprocess_data 现在返回更多东西 (val_texts_raw, val_labels_raw)
    train_dataset, val_dataset, val_texts_raw, val_labels_raw = load_and_preprocess_data(
        args.data_path, tokenizer, args.max_input_length, args.max_target_length, args.val_size
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100, 
        pad_to_multiple_of=8 if args.fp16 else None
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        predict_with_generate=True, # 启用生成，以便 compute_metrics 可以进行解码
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3, # 增加保存检查点的数量，保留更多历史最佳模型
        load_best_model_at_end=True,
        metric_for_best_model="f1_soft", # ***核心改进***：使用 f1_soft 作为最佳模型指标
        greater_is_better=True, # F1 越高越好
        gradient_accumulation_steps=args.gradient_accumulation_steps, 
        fp16=args.fp16,
        report_to="none", # 默认为 "none"，如果你要用 wandb，这里需要改成 "wandb"
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=True, # 启用梯度检查点以节省内存
        # generation_max_length=args.max_target_length, # 明确设置生成长度，以防万一
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # ***核心改进***：传入自定义的 compute_metrics
        compute_metrics=lambda p: compute_metrics(p, tokenizer, val_texts_raw, val_labels_raw),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)] if args.early_stopping_patience > 0 else []
    )

    logger.info("开始训练...")
    trainer.train(resume_from_checkpoint="./results_quad_extraction_f1_optimized_bs16_eval32/checkpoint-2200")

    logger.info("训练完成。正在保存模型...")
    trainer.save_model(args.output_dir) # 确保最佳模型被保存到 output_dir 根目录
    tokenizer.save_pretrained(args.output_dir)

    logger.info(f"模型和分词器已保存到 {args.output_dir}")

    # 保存参数以供参考
    with open(os.path.join(args.output_dir, "training_args.json"), "w", encoding='utf-8') as f:
        args_dict_to_save = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        json.dump(args_dict_to_save, f, indent=4, ensure_ascii=False)

    return args.output_dir, args.data_path, args.max_input_length, args.max_target_length, args.val_size, val_texts_raw, val_labels_raw

# 如果在命令行运行，则执行 main()
if __name__ == "__main__":
    main()