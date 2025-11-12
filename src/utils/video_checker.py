import os
import glob
import decord
import numpy as np
import traceback
from typing import Dict, List, Tuple

# 假设视频文件都在这个目录下
VIDEO_DIR = "/data/datasets/ECVA/videos"
# 要检查的文件扩展名
VIDEO_EXTENSIONS = ['*.mp4', '*.avi', '*.mov', '*.flv', '*.webm']

def check_video_decord(video_path: str) -> Tuple[bool, str]:
    """
    尝试使用 decord.VideoReader 读取视频，并返回成功/失败状态和原因。
    """
    
    # 失败原因的默认描述
    fail_reason = "Unknown Error"
    
    try:
        # 尝试初始化 VideoReader
        # 这是最可能发生 "cannot find video stream" 错误的地方
        vr = decord.VideoReader(video_path)
        
        # 进一步检查：确保可以获取总帧数和第一帧
        # 即使 VideoReader 初始化成功，尝试读取数据时也可能失败（例如视频在中间损坏）
        total_frames = len(vr)
        if total_frames == 0:
            fail_reason = "Decord successfully initialized but reported 0 total frames."
            return False, fail_reason
            
        # 尝试读取第一帧，确保解码过程正常
        _ = vr[0].asnumpy()
        
        # 成功读取
        return True, "Success"

    except decord.DECORDError as e:
        # 捕获 decord 自己的错误类型
        # 例如: ERROR cannot find video stream with wanted index: -1
        fail_reason = f"DECORDError: Stream/Codec issue. Message: {e}"
        return False, fail_reason

    except RuntimeError as e:
        # 捕获更通用的运行时错误，通常是底层 FFmpeg 抛出的
        # 尽管 `decord` 很多时候会将底层错误包装成 `DECORDError`，但捕获通用 RuntimeError 也是安全的
        if "cannot find video stream" in str(e):
             fail_reason = f"RuntimeError: Cannot find video stream (Likely corrupted/unsupported file header or codec). Message: {e}"
        else:
             fail_reason = f"RuntimeError: General decoding/runtime error. Message: {e}"
        return False, fail_reason
        
    except Exception as e:
        # 捕获所有其他意外错误，比如文件I/O错误，或者OOM（内存不足，但这个通常会Killed进程而不是抛异常）
        fail_reason = f"Unhandled Exception: {type(e).__name__}. Message: {e}"
        return False, fail_reason

def scan_videos(directory: str, extensions: List[str]) -> Dict[str, List[Dict[str, str]]]:
    """
    扫描目录下的所有视频，并使用 check_video_decord 进行检测。
    """
    results = {
        "SUCCESS": [],
        "FAILURE": [],
    }
    
    # 查找所有匹配扩展名的文件
    video_files = []
    for ext in extensions:
        # 使用 os.path.join 确保路径正确，并递归查找
        video_files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))

    if not video_files:
        print(f"Warning: No video files found in '{directory}' with extensions {extensions}")
        return results

    print(f"Found {len(video_files)} video files. Starting check...")

    for i, video_path in enumerate(video_files):
        print(f"[{i+1}/{len(video_files)}] Checking: {video_path}...")
        
        is_successful, reason = check_video_decord(video_path)
        
        filename = os.path.basename(video_path)
        
        if is_successful:
            results["SUCCESS"].append({"filename": filename, "path": video_path})
            print(f"  -> SUCCESS: Total frames ({len(decord.VideoReader(video_path))})")
        else:
            results["FAILURE"].append({"filename": filename, "path": video_path, "reason": reason})
            print(f"  -> FAILURE! Reason: {reason}")
            
    return results

def print_report(results: Dict[str, List[Dict[str, str]]]):
    """
    打印检测报告。
    """
    print("\n" + "="*50)
    print("      🎥 Video Readability Report (Decord) 🎥")
    print("="*50)
    
    total_videos = len(results["SUCCESS"]) + len(results["FAILURE"])
    
    print(f"Total Videos Checked: {total_videos}")
    print(f"✅ Successfully Read: {len(results['SUCCESS'])} videos")
    print(f"❌ Failed to Read:     {len(results['FAILURE'])} videos")
    print("="*50)

    if results["FAILURE"]:
        print("\n--- ❌ Failed Videos Details ---")
        for fail_info in results["FAILURE"]:
            print(f"\nFile: {fail_info['filename']}")
            print(f"Path: {fail_info['path']}")
            print(f"Reason: {fail_info['reason']}")
        print("-------------------------------\n")

if __name__ == "__main__":
    # 创建一个测试文件夹（如果不存在）
    if not os.path.exists(VIDEO_DIR):
        os.makedirs(VIDEO_DIR)
        print(f"Test directory '{VIDEO_DIR}' created. Please place video files inside it.")
        
    # 运行检测
    report = scan_videos(VIDEO_DIR, VIDEO_EXTENSIONS)
    
    # 打印报告
    print_report(report)