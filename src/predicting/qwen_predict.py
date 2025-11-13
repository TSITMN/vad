from transformers import AutoModelForImageTextToText, AutoProcessor ,StoppingCriteria, StoppingCriteriaList
from qwen_vl_utils import process_vision_info
from src.utils.KeywordsStoppingCriteria import KeywordsStoppingCriteria

model_path = (
    "/data/models/Qwen/Qwen3-VL-4B-Thinking"
)

model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)

processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

prompt = '''
**Output Format:** 
Your entire response **MUST** be enclosed within two specific tags: `<thinking>` and `<answer>`. 
You **MUST** use exactly one complete pair of these tags. No other instances of the tags <thinking> or <answer> are allowed.
You **MUST** start your response with <thinking>
1.  **<thinking>...</thinking> (The Reasoning Process):**
    * This section is for your internal reasoning, step-by-step thinking, analysis of the prompt, and any self-correction.
    * Number of thinking step must less than 5.
    * This must contain detailed steps on how you arrived at the final answer.
    * This section is **not** the final output for the user.

2.  **<answer>...</answer> (The Final Output):**
    * This section contains the final, concise, and direct answer to the user's request.
    * This is the only content intended for the user.
    * Do **NOT** include any extraneous text, greetings, or explanations outside of the `<thinking>` or `<answer>` tags.

    **Answer Format**
    Answer part **MUST** be JSON format
    {
        {
            "description": ""#description of abnormal event_1,
            "interval":(start_time_1 , end_time_1)
        },
        ...
        {
            "description": #description of abnormal event_n,
            "interval":(start_time_n , end_time_n)
        }
    }

**Task:** 
You are provided with a video. Your task is to analyze the video and identify any abnormal events (interval and description). 

**Example**
<thinking>1.This videos shows... ...5.Let's conclude...</thinking>
<answer>{{"description":"A person shows...","interval":(start_time , end_time)}}</answer>
'''


messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "/data/datasets/ECVA/videos/338.mp4",
                "min_pixels": 4 * 32 * 32,
                "max_pixels": 128 * 32 * 32,
                "total_pixels": 20480 * 32 * 32,
            },
            {
                "type": "text",
                "text": prompt,
            },
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
images, videos, video_kwargs = process_vision_info(
    messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
)
if videos:
    print(f"Video split into {len(videos)} segments.")

# split the videos and according metadatas
if videos is not None:
    videos, video_metadatas = zip(*videos)
    videos, video_metadatas = list(videos), list(video_metadatas)
else:
    video_metadatas = None

# since qwen-vl-utils has resize the images/videos, \
# we should pass do_resize=False to avoid duplicate operation in processor!
inputs = processor(
    text=text,
    images=images,
    videos=videos,
    video_metadata=video_metadatas,
    return_tensors="pt",
    do_resize=False,
    **video_kwargs 
)

inputs = inputs.to(model.device)


# 1. 关键：获取初始输入的长度。用于告诉 StoppingCriteria 只检查新生成的 token。
initial_input_ids_len = inputs.input_ids.shape[1] 

# 2. 定义停止关键词和获取其 IDs。
# 设置 </answer> 为主要停止标记，确保模型在完成答案后立即停止。
# 设置 <thinking> 作为次要停止标记，防止重复的 Thinking block 开始。
stop_keywords = ['</answer>', '<thinking>'] 
stop_keywords_ids = [
    # 确保每个关键词都被完整编码为一个 Tensor 序列
    processor.tokenizer.encode(keyword, add_special_tokens=False, return_tensors='pt')[0]
    for keyword in stop_keywords
]

# 3. 实例化 StoppingCriteriaList。
stopping_criteria = StoppingCriteriaList([
    KeywordsStoppingCriteria(
        keywords_ids=stop_keywords_ids, 
        input_ids_len=initial_input_ids_len # 传入初始长度
    )
])

# Inference: Generation of the output
generated_ids = model.generate(
    **inputs, 
    max_new_tokens=4096,
    do_sample=False,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0,
    num_beams=1,
    repetition_penalty=1.05,
    stopping_criteria=stopping_criteria,
)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
