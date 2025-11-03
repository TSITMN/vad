import json
import os
from torch.utils.data import Dataset

class SVTADataLoader(Dataset):
    """
    一个用于加载 SVTA 数据集（JSON Lines 格式）的 PyTorch Dataset 类。

    每一行被解析为一个 JSON 对象，包含视频信息和标题。
    这个加载器将数据转换为模型可以接受的格式。
    """
    def __init__(self, data_path, video_root, prompt="Describe this video.", max_pixels=360*420, fps=1.0):
        """
        初始化 SVTADataLoader。

        Args:
            data_path (str): 数据文件（.jsonl）的路径。
            video_root (str): 视频文件所在的根目录。
            prompt (str): 用于模型输入的文本提示。
            max_pixels (int): 视频帧的最大像素数。
            fps (float): 每秒帧数。
        """
        self.video_root = video_root
        self.prompt = prompt
        self.max_pixels = max_pixels
        self.fps = fps
        
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    self.data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line.strip()}")

    def __len__(self):
        """
        返回数据集中的样本数量。
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取并返回数据集中指定索引的样本。

        Args:
            idx (int): 样本的索引。

        Returns:
            dict: 一个包含处理后数据的字典，格式类似于多模态模型的输入。
        """
        item = self.data[idx]
        
        video_path = os.path.join(self.video_root, item['video'])
        
        # 使用第一个标题作为描述
        description = item['captions'][0] if item['captions'] else "No description available."

        # 构造类似于 messages 的格式
        # 注意：实际的视频加载和预处理通常在此处或 collate_fn 中完成
        # 这里我们返回准备好的信息
        sample = {
            "video_id": item['video_id'],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": f"{video_path}",
                            "max_pixels": self.max_pixels,
                            "fps": self.fps,
                        },
                        {
                            "type": "text",
                            "text": self.prompt,
                        },
                    ],
                },
            ]
        }
        
        return sample

