import os
import glob
import decord
import shutil
import subprocess
from typing import Dict, List, Tuple, Union

# ==============================================================================
#                                配置部分
# ==============================================================================

# ** 请根据您的实际路径修改这里 **
VIDEO_DIR = "/data/datasets/ECVA/videos"
VIDEO_EXTENSIONS = ['*.mp4']
FFMPEG_FIX_DIR = os.path.join(VIDEO_DIR, "fixed_videos")     # 存放修复后文件
FAILED_ORIGINALS_DIR = os.path.join(VIDEO_DIR, "failed_originals") # 存放原始问题文件

# ==============================================================================
#                              DECORD 检测函数
# ==============================================================================

def check_video_decord(video_path: str) -> Tuple[bool, str]:
    """
    尝试使用 decord.VideoReader 读取视频，并返回成功/失败状态和原因。
    """
    fail_reason = "Unknown Error"
    
    try:
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        
        if total_frames == 0:
            fail_reason = "Decord successfully initialized but reported 0 total frames."
            return False, fail_reason
            
        # 尝试读取第一帧，确保解码过程正常
        _ = vr[0].asnumpy() 
        
        return True, "Success"

    except decord.DECORDError as e:
        fail_reason = f"DECORDError: Stream/Codec issue. Message: {e}"
        return False, fail_reason

    except RuntimeError as e:
        if "cannot find video stream" in str(e):
             fail_reason = f"RuntimeError: Cannot find video stream (Likely corrupted header/codec). Message: {e}"
        else:
             fail_reason = f"RuntimeError: General decoding/runtime error. Message: {e}"
        return False, fail_reason
        
    except Exception as e:
        fail_reason = f"Unhandled Exception: {type(e).__name__}. Message: {e}"
        return False, fail_reason

# ==============================================================================
#                              工作流核心函数
# ==============================================================================

def scan_videos(directory: str, extensions: List[str]) -> Dict[str, List[Dict[str, str]]]:
    """扫描目录下的所有视频，并进行初始检测。"""
    
    # ... (scan_videos 函数体与之前保持一致) ...
    results = {"SUCCESS": [], "FAILURE": []}
    video_files = []
    
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(directory, ext), recursive=False)) 

    print(f"\n--- 1. 视频初次扫描 ---")
    if not video_files:
        print(f"Warning: No video files found in '{directory}' with extensions {extensions}")
        return results

    print(f"Found {len(video_files)} video files. Starting check...")

    for i, video_path in enumerate(video_files):
        filename = os.path.basename(video_path)
        print(f"[{i+1}/{len(video_files)}] Checking: {filename}...")
        
        is_successful, reason = check_video_decord(video_path)
        info = {"filename": filename, "path": video_path}
        
        if is_successful:
            results["SUCCESS"].append(info)
        else:
            info["reason"] = reason
            results["FAILURE"].append(info)
            print(f"  -> ❌ FAILURE! Reason: {reason}")
            
    return results

def run_ffmpeg_repair(failed_videos: List[Dict[str, str]], output_dir: str):
    """
    使用 subprocess 运行 FFmpeg 重新编码命令来修复视频。
    """
    print(f"\n--- 2. 自动化 FFmpeg 修复开始 ---")
    
    # ** 关键：程序化创建输出目录 **
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")

    total_failed = len(failed_videos)
    repaired_count = 0
    
    for i, info in enumerate(failed_videos):
        input_file = info['path']
        filename = info['filename']
        output_file = os.path.join(output_dir, filename)
        
        print(f"[{i+1}/{total_failed}] Fixing {filename}...")
        
        # 使用列表形式构建命令，更安全，不需要 shell=True
        # 使用重新编码 (libx264) 来解决兼容性问题，而非简单的复制
        command = [
            'ffmpeg', 
            '-y', # 自动覆盖输出文件
            '-i', input_file, 
            '-c:v', 'libx264', '-crf', '23', '-pix_fmt', 'yuv420p', # H.264 视频设置
            '-c:a', 'aac', '-b:a', '128k', # AAC 音频设置 (如果存在音频流)
            output_file
        ]
        
        try:
            # 执行命令，隐藏输出（stdout, stderr）以保持控制台整洁
            result = subprocess.run(
                command, 
                check=True, # 如果返回非零状态码，则抛出 CalledProcessError
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            repaired_count += 1
            print(f"  -> ✅ Repaired successfully.")
        except FileNotFoundError:
            print(f"  -> ❌ Error: FFmpeg command not found. Please ensure FFmpeg is in your system PATH.")
            break
        except subprocess.CalledProcessError as e:
            # 捕获 FFmpeg 内部错误，打印错误日志供调试
            print(f"  -> ❌ FFmpeg failed for {filename}. Exit Code: {e.returncode}")
            # print(f"     FFmpeg Error Output:\n{e.stderr.decode('utf-8')[:500]}...") # 打印错误信息
        except Exception as e:
            print(f"  -> ❌ An unexpected error occurred: {e}")
            
    print(f"\n--- 自动化修复完成：成功修复 {repaired_count} 个视频 ---")


def verify_repaired_videos(failed_videos: List[Dict[str, str]], repaired_dir: str):
    """验证修复后的视频文件是否能正常读取。"""
    
    print(f"\n--- 3. 验证修复后的视频 ---")
    verification_report = {"SUCCESS": [], "FAILURE": []}
    
    # 只需要检查原始失败列表中的文件在修复目录是否存在并可读
    for info in failed_videos:
        original_filename = info['filename']
        repaired_path = os.path.join(repaired_dir, original_filename)
        
        print(f"Checking repaired file: {original_filename}...")
        
        if not os.path.exists(repaired_path):
            verification_report["FAILURE"].append({
                "filename": original_filename, 
                "original_path": info['path'],
                "reason": f"Repaired file not found."
            })
            continue

        is_successful, reason = check_video_decord(repaired_path)
        
        if is_successful:
            verification_report["SUCCESS"].append({"filename": original_filename, "path": repaired_path, "original_path": info['path']})
            print(f"  -> ✅ Verification SUCCESS!")
        else:
            # 修复后仍失败，记录新的错误原因
            verification_report["FAILURE"].append({"filename": original_filename, "reason": reason, "original_path": info['path']})
            print(f"  -> ❌ Verification FAILED! New Reason: {reason}")
            
    return verification_report

def finalize_repair_and_cleanup(verification_results: Dict[str, List[Dict[str, str]]], fixed_dir: str, failed_dir: str):
    """
    将修复成功的视频移回原目录并重命名，将原始问题视频移入隔离文件夹。
    """
    
    print(f"\n--- 4. 文件清理与替换操作 ---")
    
    # 确保隔离文件夹存在
    os.makedirs(failed_dir, exist_ok=True)
    print(f"Created isolation directory: {failed_dir}")
        
    success_count = 0
    
    # 1. 处理修复成功的视频
    for info in verification_results["SUCCESS"]:
        filename = info['filename']
        original_path = info['original_path']
        repaired_path = info['path'] # 位于 FFMPEG_FIX_DIR
        
        isolated_path = os.path.join(failed_dir, filename)
        final_target_path = original_path # 最终目标是回到原路径

        try:
            # A. 隔离原始问题文件 (Move Original)
            if os.path.exists(original_path):
                 shutil.move(original_path, isolated_path)
                 print(f"  > Original failed file isolated: {filename}")
            else:
                 print(f"  > Warning: Original file {filename} not found for isolation. Skipping move.")
                 
            # B. 移动/重命名修复后的文件到原位置 (Replace with Repaired)
            # 这实现了 "将 reencoded 视频命名成原本视频的名字" 的要求
            shutil.move(repaired_path, final_target_path) 
            print(f"  > Repaired file moved and renamed to replace original: {filename}")
            
            success_count += 1
            
        except Exception as e:
            print(f"  > ❌ Error during file move/rename for {filename}: {e}")
            
    # 2. 清理 fixed_videos 文件夹中未成功的残留文件
    print(f"\n✅ 成功替换并隔离了 {success_count} 个视频文件。")
    print(f"原始问题文件现在位于: {failed_dir}")


# ==============================================================================
#                                主执行流程
# ==============================================================================

if __name__ == "__main__":
    if not os.path.exists(VIDEO_DIR):
        print(f"Error: Video directory '{VIDEO_DIR}' not found. Please verify the path.")
    else:
        # 1. 初始检测
        initial_results = scan_videos(VIDEO_DIR, VIDEO_EXTENSIONS)
        failed_videos = initial_results["FAILURE"]
        total_failed = len(failed_videos)

        print("\n" + "="*60)
        print(f"🎉 初步检测完成: 成功 {len(initial_results['SUCCESS'])} 个, 失败 {total_failed} 个")
        print("="*60)

        if total_failed > 0:
            # 2. 自动化 FFmpeg 修复
            run_ffmpeg_repair(failed_videos, FFMPEG_FIX_DIR)
            
            # 3. 验证修复后的视频
            verification_results = verify_repaired_videos(failed_videos, FFMPEG_FIX_DIR)
            
            # 4. 文件清理与替换
            finalize_repair_and_cleanup(verification_results, FFMPEG_FIX_DIR, FAILED_ORIGINALS_DIR)
            
            # 5. 最终报告
            total_verified = len(failed_videos)
            verified_success = len(verification_results["SUCCESS"])
            verified_failed = len(verification_results["FAILURE"])
            
            print("\n" + "=="*30)
            print("      ✨ 自动化修复与验证报告 ✨")
            print("=="*30)
            print(f"目标修复视频数量: {total_failed}")
            print(f"✅ 成功修复并替换原文件: {verified_success} 个")
            print(f"❌ 修复后仍失败 (未替换): {verified_failed} 个")
            
            if verified_failed > 0:
                print("\n--- ❗ 仍无法读取的视频 (位于原目录，需手动处理) ---")
                for fail in verification_results["FAILURE"]:
                    print(f"\nFile: {fail['filename']}")
                    print(f"Original Path: {fail['original_path']}")
                    print(f"Reason: {fail['reason']}")
            
            print("=="*30)
        else:
            print("所有视频均已成功读取，无需修复。")