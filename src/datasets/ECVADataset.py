import torch
import json
import os
import pandas as pd  # 引入 pandas 库
import ast         # 用于安全地将字符串转换回列表
from torch.utils.data import Dataset
from transformers import AutoProcessor
# 确保 qwen_vl_utils.py 在您的工作目录中或 Python 路径中
from qwen_vl_utils import process_vision_info
import re

class ECVADataset(Dataset):
    def __init__(
        self, 
        annotations_file, 
        processor, 
        video_root_path="/data/datasets/ECVA/videos",
        prompt_type="single_round" ,
        prompt_path="src/prompt/prompt.txt",
        is_train=True,
        ):
        """
        初始化数据集。
        Args:
            annotations_file (str): 包含标注数据的 XLSX 文件路径。
            processor: Hugging Face 的 AutoProcessor 实例。
            video_root_path (str): 存放视频文件的根目录。
            prompt_type (str): 提示类型 qa_pair , single_round
        """
        # 使用 pandas 读取 Excel 文件。
        # 默认情况下，read_excel 会自动将文件的第一行作为列名 (header)。
        df = pd.read_excel(annotations_file)
        
        # 预处理 DataFrame，将可能的空值 (NaN) 替换为空字符串，防止后续处理出错
        df.fillna('', inplace=True)
        
        self.annotations_df = df
        self.processor = processor
        self.video_root_path = video_root_path
        self.prompt_type = prompt_type
        self.prompt_path = prompt_path
        self.is_training = is_train

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

        # 1. 获取视频路径 (使用索引 0)
        video_id = str(item.iloc[0]) # 索引 0 对应 'video_id'
        video_path = os.path.join(self.video_root_path, f"{video_id}.mp4")
        # print(video_path)
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
            print(f"Warning: Failed to parse key sentences for video_id {video_id}")
        # 3. 处理A3,moment列，提取时间戳信息（使用索引4）
        moment_str = item.iloc[4]  # 索引4对应 'A3,moment'
        moment_str_safe = re.sub(r'(\d+)', r'"\1"', str(moment_str))
        #[0005,0011],[0022,0027] --> [(5,11),(22,27)]

        moment_list = []
        try:
            moment_pairs = [ast.literal_eval(moment_str_safe)] if moment_str_safe else [[]]
            for pair in moment_pairs:
                if isinstance(pair, list) and len(pair) == 2:
                    start, end = int(pair[0]), int(pair[1])
                    moment_list.append( (start, end) )
        except (ValueError, SyntaxError):
            moment_list = []
            print(f"Warning: Failed to parse moment for video_id {video_id}")
        
        # print(f"Debug: moment_list for video_id {video_id}: {moment_list}") 
        add_gen_prompt = not self.is_training
        
        if self.prompt_type == "single_round":
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read()
                
            user_msg = {
                    "role": "user",
                    "content": [
                        {
                            "type": "video", 
                            "video": video_path, 
                            "min_pixels": 4 * 32 * 32,
                            "max_pixels": 32 * 32 * 32,
                            "total_pixels": 24576 * 32 * 32,
                        },
                        {
                            "type": "text",  
                            "text": prompt
                        }
                    ]
                }
            assistant_msg = {
                    "role": "assistant",
                    "content":{
                        "description": item.iloc[5],  # 索引 5 对应 'A3 - description'
                        "intervals": moment_list
                    }
            }
            messages = [user_msg , assistant_msg]
            return {
                    "messages": messages,
                    "text": self.processor.apply_chat_template(messages , 
                                                               tokenize=False ,
                                                               add_generation_prompt=add_gen_prompt),
                    "moment_list": moment_list
                    }
        
        if self.prompt_type == "qa_pair":   
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
                    {
                        "type": "video", 
                        "video": video_path, 
                        "min_pixels": 4 * 32 * 32,
                        "max_pixels": 128 * 32 * 32,
                        "total_pixels": 20480 * 32 * 32,
                        "nframes": 2 ,
                    },
                    {
                        "type": "text",  
                        "text": first_question
                    }
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

            return {
                    "messages": messages,
                    "text": self.processor.apply_chat_template(messages , tokenize=False ,add_generation_prompt=add_gen_prompt),
                    }
    
if __name__ == "__main__":
    # 测试数据加载器
    processor = AutoProcessor.from_pretrained("/data/models/Qwen/Qwen3-VL-8B-Thinking")
    dataset = ECVADataset(
        annotations_file="/data/datasets/ECVA/Video_Annotation.xlsx",
        processor=processor,
        video_root_path="/data/datasets/ECVA/videos"
    )
    
    # 获取一个样本进行测试
    sample = dataset[0]
    
    # 格式化输出json以便查看
    print(json.dumps(sample, indent=4, ensure_ascii=False))

