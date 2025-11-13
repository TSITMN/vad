import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import logging
import io

from src.datasets.ECVADataset import ECVADataset

logger = logging.getLogger(__name__)

def _process_and_collate_sub_batch(sub_batch, processor):
    """处理单个子批次的辅助函数"""
    if not sub_batch:
        return None
    
    all_messages = [item["messages"] for item in sub_batch]
    all_chats = [item["text"] for item in sub_batch]
    video_paths = [item["messages"][0]["content"][0]["video"] for item in sub_batch]
    
    images, videos, video_kwargs = process_vision_info(
        all_messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
    )
    
    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
    else:
        video_metadatas = None
    
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
    
    del all_messages, all_chats, images, videos, video_metadatas, video_kwargs
    
    labels = model_inputs["input_ids"].clone()
    
    if model_inputs["input_ids"].shape[0] > 0:
        return {"video_paths": video_paths, **model_inputs, "labels": labels}
    else:
        return None


def custom_collate_fn(batch_list, processor, max_total_size_gb=2.0):
    """
    根据视频文件总大小动态拆分批次的 collate 函数。
    
    Args:
        batch_list: 原始批次数据列表
        processor: 处理器
        max_total_size_gb: 单个批次允许的最大总文件大小(GB)
    
    Returns:
        list: 拆分后的子批次列表
    """
    max_total_size_bytes = max_total_size_gb * (1024 ** 3)
    sub_batches = []
    current_sub_batch = []
    current_total_size = 0
    
    for item in batch_list:
        video_size = item.get("video_size", 0)
        
        # 如果添加当前视频会超过限制,且当前子批次不为空,则开始新的子批次
        if current_sub_batch and current_total_size + video_size > max_total_size_bytes:
            # 处理当前子批次
            processed_batch = _process_and_collate_sub_batch(current_sub_batch, processor)
            if processed_batch:
                sub_batches.append(processed_batch)
                logger.info(f"Created sub-batch with {len(current_sub_batch)} videos, total size: {current_total_size / (1024**3):.2f} GB")
            
            # 开始新的子批次
            current_sub_batch = [item]
            current_total_size = video_size
        else:
            # 添加到当前子批次
            current_sub_batch.append(item)
            current_total_size += video_size
    
    # 处理最后一个子批次
    if current_sub_batch:
        processed_batch = _process_and_collate_sub_batch(current_sub_batch, processor)
        if processed_batch:
            sub_batches.append(processed_batch)
            logger.info(f"Created sub-batch with {len(current_sub_batch)} videos, total size: {current_total_size / (1024**3):.2f} GB")
    
    return sub_batches


class VariableBatchDataLoader(DataLoader):
    """
    支持动态批次拆分的 DataLoader。
    当 collate_fn 返回列表时,逐个产出子批次。
    """
    def __iter__(self):
        for batch_list in super().__iter__():
            if isinstance(batch_list, list) and batch_list:
                # 如果返回的是批次列表,逐个产出
                yield from batch_list
            elif batch_list:
                # 如果返回的是单个批次,直接产出
                yield batch_list


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
    max_total_size_gb=2.0,  # 新增参数:单个批次最大总文件大小(GB)
):
    """
    创建一个用于 ECVA 数据集的 DataLoader。

    Args:
        annotations_file (str): 包含标注数据的 XLSX 文件路径。
        video_root_path (str): 存放视频文件的根目录。
        processor: Hugging Face 的 AutoProcessor 实例,将用作 collator。
        batch_size (int): 每个批次的大小。
        num_workers (int): 用于数据加载的子进程数。
        shuffle (bool): 是否在每个 epoch 开始时打乱数据。
        sample_size (int, optional): 如果提供,则只加载指定数量的样本。默认为 None。
        start_index (int): 数据集的起始索引,用于断点续传。
        max_total_size_gb (float): 单个批次允许的最大视频文件总大小(GB)。

    Returns:
        VariableBatchDataLoader: 配置好的 DataLoader 实例。
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
        dataset = Subset(dataset, range(min(sample_size, len(dataset))))
        print(f"Using a subset of {len(dataset)} samples for testing.")

    # 2. 创建支持动态批次的 DataLoader
    dataloader = VariableBatchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: custom_collate_fn(batch, processor, max_total_size_gb=max_total_size_gb),
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
        max_total_size_gb=1.5,  # 测试文件大小限制
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

