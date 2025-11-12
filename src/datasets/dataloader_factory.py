import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import logging
import io
# from unittest.mock import patch # 不再需要

# 从我们之前修改的文件中导入 ECVADataset 类
# 假设该文件名为 ECVADataLoader.py
from src.datasets.ECVADataset import ECVADataset

logger = logging.getLogger(__name__)

def custom_collate_fn(batch_list, processor):
    # 1. 收集批次中所有样本的 messages 和 text
    all_messages = [item["messages"] for item in batch_list]
    all_chats = [item["text"] for item in batch_list]
    # 提取视频路径
    video_paths=[item["messages"][0]["content"][0]["video"] for item in batch_list]
    # 2. 一次性为整个批次预处理视觉信息
    # process_vision_info 可以直接处理 messages 列表
    images, videos, video_kwargs = process_vision_info(
        all_messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
    )
    # split the videos and according metadatas
    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
    else:
        video_metadatas = None
    # 3. 将所有样本合并为一个批次
    # 将 process_vision_info 的输出直接传递给 processor
    model_inputs = processor(
        text=all_chats,
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        return_tensors="pt",
        padding=True,
        truncation=True,
        **video_kwargs,
    )

    # ✅ 优化点 2: 删除不再需要的中间变量
    del all_messages, all_chats, images, videos, video_metadatas, video_kwargs

    # 4. 创建标签
    labels = model_inputs["input_ids"].clone()
    # print(processor.tokenizer.decode(model_inputs["input_ids"][0] , skip_special_tokens=True))
    
    # 如果批次中仍有有效样本，则返回它们
    if model_inputs["input_ids"].shape[0] > 0:
        return {"video_paths": video_paths, **model_inputs, "labels": labels}
    else:
        return {}


def create_ecva_dataloader(
    annotations_file,
    video_root_path,
    processor,
    batch_size=2,
    num_workers=4,
    shuffle=True,
    prompt_type="single_round",
    sample_size=None,
    is_train=True,
    start_index=0,
):
    """
    创建一个用于 ECVA 数据集的 DataLoader。

    Args:
        annotations_file (str): 包含标注数据的 XLSX 文件路径。
        video_root_path (str): 存放视频文件的根目录。
        processor: Hugging Face 的 AutoProcessor 实例，将用作 collator。
        batch_size (int): 每个批次的大小。
        num_workers (int): 用于数据加载的子进程数。
        shuffle (bool): 是否在每个 epoch 开始时打乱数据。
        sample_size (int, optional): 如果提供，则只加载指定数量的样本。默认为 None。
        start_index (int): 数据集的起始索引，用于断点续传。

    Returns:
        torch.utils.data.DataLoader: 配置好的 DataLoader 实例。
    """
    # 1. 实例化我们之前创建的 Dataset
    full_dataset = ECVADataset(
        annotations_file=annotations_file,
        processor=processor,
        video_root_path=video_root_path,
        prompt_type=prompt_type,
        is_train=is_train,
    )

    dataset = full_dataset
    
    # 首先根据 start_index 切片
    if start_index > 0:
        if start_index >= len(dataset):
            print(f"Warning: start_index ({start_index}) is out of bounds for dataset size ({len(dataset)}). Returning an empty dataloader.")
            dataset = Subset(dataset, [])
        else:
            dataset = Subset(dataset, range(start_index, len(dataset)))
            print(f"Sliced dataset: starting from index {start_index}. New size: {len(dataset)}")

    # 然后在切片后的数据集上应用 sample_size
    if sample_size is not None:
        # 如果指定了 sample_size，则创建一个只包含前 N 个样本的子集
        dataset = Subset(dataset, range(min(sample_size, len(dataset))))
        print(f"Using a subset of {len(dataset)} samples for testing.")

    # 2. 创建 DataLoader
    #    - collate_fn=processor: 这是关键。processor 知道如何将批次中的
    #      多个样本（文本和视频）正确地填充和堆叠成一个批次张量。
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: custom_collate_fn(batch, processor),
        pin_memory=True,
    )
    
    return dataloader

if __name__ == '__main__':
    # --- 测试代码 ---
    # 这个例子展示了如何在训练脚本中使用这个工厂函数

    # 1. 加载 Processor
    print("正在加载 Processor...")
    processor = AutoProcessor.from_pretrained("/data/models/Qwen/Qwen3-VL-8B-Thinking")
    processor.tokenizer.padding_side = 'left' # 设置为左填充

    # 2. 使用工厂函数创建 DataLoader
    print("正在创建 DataLoader...")
    train_dataloader = create_ecva_dataloader(
        annotations_file="/data/datasets/ECVA/Video_Annotation.xlsx",
        video_root_path="/data/datasets/ECVA/videos",
        processor=processor,
        batch_size=2,  # 设置批次大小为 2
        num_workers=0, # 在主进程中加载数据以便于调试
        shuffle=False,  # 关闭打乱以便于复现
        prompt_type="single_round",
        sample_size=5, # 添加采样测试
    )

    # 3. 从 DataLoader 中获取一个批次的数据
    print("正在从 DataLoader 中获取一个批次...")
    print(type(train_dataloader))
    print(train_dataloader.__dict__.keys())
    # try:
    first_batch = next(iter(train_dataloader))

    # 4. 打印批次数据的类型和形状，验证其正确性
    print("\n--- 批次数据检查 ---")
    print(f"批次类型: {type(first_batch)}")
    print("批次中的键:", first_batch.keys())
    
    for key, value in first_batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  - {key}: Tensor, shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  - {key}: {type(value)}")

