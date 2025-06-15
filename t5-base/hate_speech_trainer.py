# hate_speech_trainer.py

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

# 设置日志 (在 Jupyter 中可能会有重复日志，但为了独立运行保留)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义有效的目标群体
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
    """
    target = quad.get("target", "NULL")
    argument = quad.get("argument", "")
    targeted_group_raw = quad.get("targeted_group", "non-hate")
    hateful = quad.get("hateful", "non-hate")

    targeted_group = TARGETED_GROUP_MAPPING.get(targeted_group_raw, targeted_group_raw)
    if targeted_group not in VALID_TARGETED_GROUPS:
        logger.warning(f"检测到无效的 targeted_group '{targeted_group_raw}'。将其映射为 'others'。原始四元组: {quad}")
        targeted_group = 'others'

    return f"{target} | {argument} | {targeted_group} | {hateful} [END]"

def create_target_string(quadruplets_list):
    """
    从四元组字典列表中创建完整的输出目标字符串。
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
        labels[labels == self.tokenizer.pad_token_id] = -100

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
    target_strings = []

    for item in raw_data:
        text = item.get("content")
        output = item.get("output")

        if text is None or output is None:
            logger.warning(f"由于缺少 'content' 或 'output'，跳过该项: {item}")
            continue
        
        texts.append(text)
        target_strings.append(output)

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
    
    return train_dataset, val_dataset, val_texts, val_labels # 返回 val_texts 和 val_labels 以便后续评估直接使用

# 修改 main 函数以接受参数字典
def main(args_dict=None):
    if args_dict is None: # 如果没有提供 args_dict，则从命令行解析
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_path", type=str, default="train.json", help="train.json 文件路径")
        parser.add_argument("--model_name_or_path", type=str, default="Langboat/mengzi-t5-base", help="预训练模型名称或路径")
        # 修改 output_dir 的默认值，指向新的目录
        parser.add_argument("--output_dir", type=str, default="./results_quad_extraction_epoch23_bs16", help="保存训练模型的目录")
        # 修改 num_train_epochs 的默认值
        parser.add_argument("--num_train_epochs", type=int, default=23, help="训练轮数")
        parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="训练批次大小")
        parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="评估批次大小")
        parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
        parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
        parser.add_argument("--max_input_length", type=int, default=128, help="最大输入序列长度")
        parser.add_argument("--max_target_length", type=int, default=64, help="生成时的最大目标序列长度")
        parser.add_argument("--val_size", type=float, default=0.1, help="用于验证的训练数据比例")
        parser.add_argument("--save_steps", type=int, default=500, help="每 X 步保存一次检查点。")
        parser.add_argument("--eval_steps", type=int, default=500, help="每 X 步评估一次。")
        parser.add_argument("--logging_steps", type=int, default=100, help="每 X 步记录一次日志。")
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
        parser.add_argument("--fp16", action="store_true", help="使用 FP16 训练")
        parser.add_argument("--early_stopping_patience", type=int, default=10, help="早停的耐心值")
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
    
    # 修改这里，load_and_preprocess_data 现在返回更多东西
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
        predict_with_generate=True, 
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2, # 保存最近的两个检查点，注意这会删除旧的，如果不想删除需要设大或为None
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", 
        greater_is_better=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps, 
        fp16=args.fp16,
        report_to="none",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)] if args.early_stopping_patience > 0 else []
    )

    logger.info("开始训练...")
    # 明确指定从哪个检查点恢复训练
    # 确保这个路径是正确的，并且 checkpoint-3000 文件夹存在于你的文件系统中
    trainer.train(resume_from_checkpoint="./results_quad_extraction_epoch23_bs16_resume/checkpoint-3500")

    logger.info("训练完成。正在保存模型...")
    trainer.save_model(args.output_dir) 
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