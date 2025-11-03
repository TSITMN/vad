import torch
import json
import os
import pandas as pd  # 引入 pandas 库
import ast         # 用于安全地将字符串转换回列表
from torch.utils.data import Dataset
from transformers import AutoProcessor
# 确保 qwen_vl_utils.py 在您的工作目录中或 Python 路径中
from qwen_vl_utils import process_vision_info

class ECVADataLoader(Dataset):
    """
    用于 ECVA 数据集的多轮对话指令微调 Dataloader。
    此版本直接从 XLSX (Excel) 文件读取标注，并使用列索引访问数据。
    """
    def __init__(self, annotations_file, processor, video_root_path="/public/home/djingwang/tychen/data/ECVA/videos"):
        """
        初始化数据集。
        Args:
            annotations_file (str): 包含标注数据的 XLSX 文件路径。
            processor: Hugging Face 的 AutoProcessor 实例。
            video_root_path (str): 存放视频文件的根目录。
        """
        # 使用 pandas 读取 Excel 文件。
        # 默认情况下，read_excel 会自动将文件的第一行作为列名 (header)。
        df = pd.read_excel(annotations_file)
        
        # 预处理 DataFrame，将可能的空值 (NaN) 替换为空字符串，防止后续处理出错
        df.fillna('', inplace=True)
        
        self.annotations_df = df
        self.processor = processor
        self.video_root_path = video_root_path

    def __len__(self):
        # 长度是 DataFrame 的行数
        return len(self.annotations_df)

    def __getitem__(self, idx):
        # 使用 .iloc[idx] 获取指定行的数据，它会返回一个 Series
        item = self.annotations_df.iloc[idx]

        # --- 通过列的整数索引访问数据 ---
        # 请确保您的 Excel 文件列顺序与以下注释一致：
        # 索引 0: 'video_id'
        # 索引 1: 'A1 Classification'
        # 索引 2: 'A2 - reason'
        # 索引 3: 'A2 - result'
        # 索引 4: 'A3,moment' (此脚本中未使用)
        # 索引 5: 'A3 - description'
        # 索引 6: 'A4 - key sentences'
        # ------------------------------------

        # 1. 获取视频路径并检查文件是否存在 (使用索引 0)
        video_id = str(item.iloc[0]) # 索引 0 对应 'video_id'
        video_path = os.path.join(self.video_root_path, f"{video_id}.mp4")

        if not os.path.exists(video_path):
            print(f"警告: 视频文件未找到 {video_path}，将加载下一个有效样本。")
            return self.__getitem__((idx + 1) % len(self))

        # 2. 处理 'A4 - key sentences' 列 (使用索引 6)
        key_sentences_str = item.iloc[6] # 索引 6 对应 'A4 - key sentences'
        try:
            # ast.literal_eval 是比 eval() 更安全的方法
            key_sentences_list = ast.literal_eval(key_sentences_str) if key_sentences_str else []
            if not isinstance(key_sentences_list, list):
                key_sentences_list = []
        except (ValueError, SyntaxError):
            # 如果解析失败，视为空列表
            key_sentences_list = []

        # 3. 构建多轮对话 (使用相应的列索引)
        qa_pairs = [
            ("请对视频中的异常行为进行分类。", item.iloc[1]),             # 索引 1 对应 'A1 Classification'
            ("导致此异常发生的原因是什么？", item.iloc[2]),             # 索引 2 对应 'A2 - reason'
            ("该异常事件导致了什么结果？", item.iloc[3]),                 # 索引 3 对应 'A2 - result'
            ("请详细描述视频中异常事件发生的关键时刻。", item.iloc[5]), # 索引 5 对应 'A3 - description'
            ("请用简短的词组概括视频中的关键动作或物体。", ", ".join(key_sentences_list))
        ]

        messages = []
        
        # --- 第一轮对话 (包含视频) ---
        first_question, first_answer = qa_pairs[0]
        user_msg_1 = {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "max_pixels": 360*420, "fps": 1.0},
                {"type": "text",  "text": first_question}
            ]
        }
        assistant_msg_1 = {"role": "assistant", "content": first_answer}
        messages.extend([user_msg_1, assistant_msg_1])

        # --- 后续的纯文本对话 ---
        for question, answer in qa_pairs[1:]:
            # 只有当回答不为空时才添加这一轮对话
            if answer and str(answer).strip():
                user_msg_n = {"role": "user", "content": question}
                assistant_msg_n = {"role": "assistant", "content": answer}
                messages.extend([user_msg_n, assistant_msg_n])
        
        # 4. 应用聊天模板
        chat = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # 5. 预处理视觉信息
        img_inputs, vid_inputs, vid_kwargs = process_vision_info(
            [user_msg_1], return_video_kwargs=True
        )

        # 6. 合并文本与视觉数据
        model_inputs = self.processor(
            text=[chat], images=img_inputs, videos=vid_inputs,
            padding="longest", return_tensors="pt", **vid_kwargs
        )
        model_inputs = {k: v.squeeze(0) for k, v in model_inputs.items()}

        # 7. 创建训练标签
        labels = model_inputs["input_ids"].clone()

        return {**model_inputs, "labels": labels}