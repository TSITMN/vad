import os
import json
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm

from src.datasets.dataloader_factory import create_ecva_dataloader
from src.utils.KeywordsStoppingCriteria import KeywordsStoppingCriteria

def convert_jsonl_to_json(jsonl_filepath, json_filepath):
    """
    将 JSONL 文件转换为结构化的 JSON 数组文件。
    """
    results = []
    print(f"Converting JSONL file '{jsonl_filepath}' to JSON file '{json_filepath}'...")
    with open(jsonl_filepath, "r", encoding="utf-8") as f_jsonl:
        for line in f_jsonl:
            # 尝试加载每一行 JSON 对象
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping malformed line in JSONL: {line.strip()}. Error: {e}")
    
    # 将完整的列表保存为 JSON 文件
    with open(json_filepath, "w", encoding="utf-8") as f_json:
        json.dump(results, f_json, indent=4, ensure_ascii=False)
    
    print(f"Conversion complete. Total {len(results)} records saved.")

def predict(
    model_path,
    annotations_file,
    video_root_path,
    output_file,
    batch_size=2,
    prompt_type="single_round",
    sample_size=None,
    start_index=0,
):
    """
    使用指定的模型对 ECVA 数据集进行预测，并将结果保存到 JSON 文件。

    Args:
        model_path (str): 预训练模型的路径。
        annotations_file (str): 包含标注数据的 XLSX 文件路径。
        video_root_path (str): 存放视频文件的根目录。
        output_file (str): 保存预测结果的 JSON 文件路径。
        batch_size (int): 推理时的批次大小。
        prompt_type (str): 使用的提示类型。
        sample_size (int, optional): 如果提供，则只预测指定数量的样本。默认为 None。
        start_index (int): 数据集的起始索引，用于断点续传。
    """
    # 1. 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. 加载模型和处理器
    print(f"Loading model and processor from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = 'left'  # 设置为左填充
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).eval()
    
    # 3. 创建 DataLoader
    print("Creating DataLoader for prediction...")
    dataloader = create_ecva_dataloader(
        annotations_file=annotations_file,
        video_root_path=video_root_path,
        processor=processor,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,  # 推理时不需要打乱
        prompt_type=prompt_type,
        sample_size=sample_size,
        is_train=False,
        start_index=start_index,
    )

    # 4. 设置输出路径 (使用 .jsonl 作为临时文件)
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    jsonl_output_file = output_file.replace('.json', '.jsonl') # 创建一个 JSONL 临时文件路径
    
    # 如果是从头开始，则清空文件
    if start_index == 0 and os.path.exists(jsonl_output_file):
        os.remove(jsonl_output_file)
        
    print(f"Saving temporary predictions to JSONL file: {jsonl_output_file}...")

    # 5. 进行推理并将结果保存到 JSONL 文件
    print("Starting prediction...")
    with torch.no_grad():
        # 使用追加模式 "a" 以支持断点续传
        with open(jsonl_output_file, "a", encoding="utf-8") as f_jsonl:
            for batch in tqdm(dataloader, desc="Predicting"):
                # 如果批次为空（例如，因为所有样本都处理失败），则跳过
                if not batch:
                    continue
                
                # 准备模型输入
                # 将张量移动到指定设备
                model_inputs = {
                    k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
                }
                
                # 从模型输入中移除 'labels'，因为它仅用于训练
                model_inputs.pop("labels", None)

                # 停止条件（如果需要，可以取消注释并启用）
                # initial_input_ids_len = model_inputs["input_ids"].shape[1]
                # stop_keywords = ['</answer>', '<thinking>'] 
                # stop_keywords_ids = [
                #     processor.tokenizer.encode(keyword, add_special_tokens=False, return_tensors='pt')[0].to(device)
                #     for keyword in stop_keywords
                # ]
                # stopping_criteria = StoppingCriteriaList([
                #     KeywordsStoppingCriteria(
                #         keywords_ids=stop_keywords_ids, 
                #         input_ids_len=initial_input_ids_len
                #     )
                # ])

                # 生成文本
                gen_kwargs = {
                    "max_new_tokens": 2048, 
                    "do_sample": False,
                    # "stopping_criteria": stopping_criteria,
                }
                pred = model.generate(**model_inputs, **gen_kwargs)

                # 解码生成的文本
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs["input_ids"], pred)
                ]
                decoded_preds = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

                # 收集本批次结果并写入 JSONL
                video_paths = batch["video_paths"]
                for video_path, prediction in zip(video_paths, decoded_preds):
                    result = {"video_path": video_path, "prediction": prediction.strip()}
                    f_jsonl.write(json.dumps(result, ensure_ascii=False) + '\n')
                
                if device == "cuda":
                    torch.cuda.empty_cache()

    # 6. 将临时 JSONL 文件转换为最终的 JSON 数组文件
    print("Prediction finished. Starting final conversion...")
    convert_jsonl_to_json(jsonl_output_file, output_file)
    
    # 可选：删除临时 JSONL 文件 (这里选择保留，以便调试)
    # os.remove(jsonl_output_file)
    # print(f"Temporary JSONL file removed: {jsonl_output_file}")
    
    print(f"Final results saved to {output_file}.")

if __name__ == "__main__":
    # --- 配置参数 ---
    MODEL_PATH = "/data/models/Qwen/Qwen3-VL-4B-Thinking"
    ANNOTATIONS_FILE = "/data/datasets/ECVA/Video_Annotation.xlsx"
    VIDEO_ROOT_PATH = "/data/datasets/ECVA/videos"
    OUTPUT_FILE = "output/predict/ecva_predict.json"
    BATCH_SIZE = 1
    PROMPT_TYPE = "single_round"
    SAMPLE_SIZE = 16 
    START_INDEX = 326
    #294 error video
    #297 298 299
    predict(
        model_path=MODEL_PATH,
        annotations_file=ANNOTATIONS_FILE,
        video_root_path=VIDEO_ROOT_PATH,
        output_file=OUTPUT_FILE,
        batch_size=BATCH_SIZE,
        prompt_type=PROMPT_TYPE,
        # sample_size=SAMPLE_SIZE,
        start_index=START_INDEX,
    )