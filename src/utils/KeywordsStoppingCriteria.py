# 文件路径: src/utils/KeywordsStoppingCriteria.py
import torch
from transformers import StoppingCriteria

class KeywordsStoppingCriteria(StoppingCriteria):
    """
    自定义停止标准：当模型生成了任一关键词序列时，停止生成。
    """
    def __init__(self, keywords_ids: list, input_ids_len: int):
        """
        :param keywords_ids: 要停止的 token ID 列表 (每个关键词是一个 Tensor)。
        :param input_ids_len: 初始输入 prompt 的 token 长度。
        """
        super().__init__()
        self.keywords_ids = [keyword.to('cuda') for keyword in keywords_ids] # 确保关键词 IDs 在 GPU 上
        self.keywords_length = [len(keyword) for keyword in keywords_ids]
        self.input_ids_len = input_ids_len
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 获取当前已生成的 token 序列 (排除初始输入 prompt 的 token)
        # 即使使用了 beam search，input_ids 的形状仍是 (num_beams, current_length)
        
        # 只需要检查第一条序列，因为停止条件通常适用于所有 beams/samples
        current_ids = input_ids[0] 
        
        # 排除 prompt 部分的长度
        generated_ids = current_ids[self.input_ids_len:]
        
        # 遍历所有停止关键词
        for keyword_ids, keyword_len in zip(self.keywords_ids, self.keywords_length):
            if generated_ids.shape[-1] >= keyword_len:
                # ！！！ 核心：检查已生成序列的末尾 N 个 token 是否匹配 ！！！
                if torch.equal(generated_ids[-keyword_len:], keyword_ids.to(input_ids.device)):
                    return True # 匹配到停止序列，返回 True
        return False