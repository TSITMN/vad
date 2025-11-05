import torch
from torch.utils.data import DataLoader, default_collate
from torch.optim import AdamW
from transformers import AutoProcessor, AutoModelForCausalLM
from tqdm import tqdm
import os
import sys
import argparse
from typing import List
from code.vad.src.datasets.ECVADataset import ECVADataLoader

# ==============================================================================
# 1. 动态批次整理函数 (Collate Function)
# ==============================================================================
def data_collator(batch: List[dict]) -> dict:
    """
    自定义批次整理函数，用于处理变长序列（如 input_ids, attention_mask）
    并保持图像数据不变。
    """
    
    # 1. 分离图像张量
    image_list = [item.pop("pixel_values") for item in batch]
    
    # 2. 对文本部分使用默认整理器（处理填充）
    collated_text = default_collate(batch) 
    
    # 3. 将图像列表重新添加到批次字典中
    collated_text["pixel_values"] = image_list 
    
    return collated_text

# ==============================================================================
# 2. 训练函数
# ==============================================================================
def train(args):
    """
    一个基础的训练函数，用于在 ECVA 数据集上微调 Qwen-VL 模型。
    """
    # --- 1. 配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 展开用户目录 ~
    model_name = os.path.expanduser(args.model_name)
    output_dir = args.output_dir
    
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")

    # --- 2. 加载处理器、数据集和数据加载器 ---
    print("Loading processor and dataset...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    train_dataset = ECVADataLoader(
        annotations_file=args.annotations_file,
        processor=processor,
        video_root_path=args.video_root_path
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=data_collator,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # --- 3. 加载模型和优化器 ---
    print(f"Loading model: {model_name}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    # ----------------------------------------------------
    # !!! 改进：分层学习率逻辑定义优化器 !!!
    # ----------------------------------------------------
    
    LLM_LR = args.learning_rate 
    VISION_LR = args.learning_rate * args.vision_lr_ratio  # 视觉编码器学习率
    CONNECT_LR = args.learning_rate * args.connect_lr_ratio # 连接层学习率
    
    param_groups = []
    
    # 遍历所有可训练参数
    trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    for name, param in tqdm(trainable_params, desc="Setting up parameter groups"):
        if "vision_tower" in name:
            lr = VISION_LR
        elif "mm_projector" in name or "mlp_projector" in name:
            lr = CONNECT_LR
        else:
            lr = LLM_LR
            
        param_groups.append({"params": param, "lr": lr})

    print(f"Optimizer Groups: LLM={LLM_LR}, Vision={VISION_LR}, Connect={CONNECT_LR}")
    optimizer = AdamW(param_groups) 
    
    # --- 4. 训练循环 ---
    print("Starting training...")
    model.train() # 明确设置为训练模式
    
    total_steps = len(train_loader) * args.epochs
    global_step = 0

    for epoch in range(args.epochs):
        print(f"--- Epoch {epoch + 1}/{args.epochs} ---")
        total_loss = 0
        
        model.zero_grad() # 在每个 epoch 开始时清零所有梯度
        
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            # 移动张量到设备
            input_batch = {}
            for k, v in batch.items():
                if k != "pixel_values" and isinstance(v, torch.Tensor):
                    input_batch[k] = v.to(device)
                elif k == "pixel_values" and isinstance(v, list):
                    # 将 List 中的每个图像张量移动到 GPU/主设备
                    input_batch[k] = [img.to(device) for img in v]
                else:
                    input_batch[k] = v 

            outputs = model(**input_batch) 
            loss = outputs.loss
            
            # 梯度累积：将损失按累积步数平均
            loss = loss / args.gradient_accumulation_steps
            
            # 反向传播
            loss.backward()
            
            total_loss += loss.item() * args.gradient_accumulation_steps # 累积原始损失值
            
            # 检查是否应该更新权重
            if (i + 1) % args.gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                model.zero_grad()
                global_step += 1
            
            # 打印损失
            if (i + 1) % (10 * args.gradient_accumulation_steps) == 0: 
                current_avg_loss = total_loss / (i + 1) 
                print(f"Step {global_step}/{total_steps // args.gradient_accumulation_steps}, Avg Loss: {current_avg_loss:.4f}, Recent Loss: {loss.item() * args.gradient_accumulation_steps:.4f}")


        # 确保最后一个不完整的累积批次也进行优化
        if (len(train_loader) % args.gradient_accumulation_steps != 0):
             optimizer.step()
             model.zero_grad()
             
        avg_loss = total_loss / len(train_loader)
        print(f"End of Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

    # --- 5. 保存模型 ---
    print("Training finished. Saving model...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    print(f"Model saved to {output_dir}")


# ==============================================================================
# 3. 参数解析和主函数
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning Qwen-VL model on ECVA dataset.")
    
    # --- 模型和路径参数 ---
    parser.add_argument(
        "--model_name",
        type=str,
        default="~/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct/",
        help="Path to the pre-trained Qwen-VL model.",
    )
    parser.add_argument(
        "--annotations_file",
        type=str,
        default="/public/home/djingwang/tychen/data/ECVA/Video_Annotation.xlsx",
        help="Path to the training annotations file.",
    )
    parser.add_argument(
        "--video_root_path",
        type=str,
        default="/public/home/djingwang/tychen/data/ECVA/videos",
        help="Root directory containing video files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/public/home/djingwang/tychen/code/VAD/outputs/models/qwen_vl_ecva_finetuned",
        help="Directory to save the fine-tuned model and processor.",
    )
    
    # --- 训练参数 ---
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Device batch size (per GPU).")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of steps to accumulate gradients before updating weights.",
    )
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Base learning rate for LLM layers.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader worker processes.")
    
    # --- 分层学习率参数 ---
    parser.add_argument(
        "--vision_lr_ratio",
        type=float,
        default=0.1,
        help="Ratio of Vision LR to Base LR (e.g., 0.1 means Vision LR is 1/10 of Base LR).",
    )
    parser.add_argument(
        "--connect_lr_ratio",
        type=float,
        default=2.0,
        help="Ratio of Connect Layer LR to Base LR (e.g., 2.0 means Connect LR is 2x Base LR).",
    )
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    train(args)