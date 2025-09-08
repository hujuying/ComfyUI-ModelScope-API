import requests
import json
import time
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import os
import folder_paths
import base64
import tempfile

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'modelscope_config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {
            "default_model": "Qwen/Qwen-Image",
            "timeout": 720,
            "image_download_timeout": 30,
            "default_prompt": "A beautiful landscape"
        }

def save_config(config: dict) -> bool:
    config_path = os.path.join(os.path.dirname(__file__), 'modelscope_config.json')
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存配置失败: {e}")
        return False

def save_api_tokens(tokens):
    """保存多个API Token"""
    tokens_path = os.path.join(os.path.dirname(__file__), '.qwen_tokens')
    try:
        with open(tokens_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(tokens))  # 每个token一行
    except Exception as e:
        print(f"保存tokens失败(.qwen_tokens): {e}")
    
    try:
        cfg = load_config()
        cfg["api_tokens"] = tokens
        if save_config(cfg):
            return True
        return False
    except Exception as e:
        print(f"保存tokens失败(config.json): {e}")
        return False

def load_api_tokens():
    """加载多个API Token"""
    tokens_path = os.path.join(os.path.dirname(__file__), '.qwen_tokens')
    try:
        cfg = load_config()
        tokens_from_cfg = cfg.get("api_tokens", [])
        if tokens_from_cfg and isinstance(tokens_from_cfg, list):
            return [token.strip() for token in tokens_from_cfg if token.strip()]
    except Exception as e:
        print(f"读取config.json中的tokens失败: {e}")
    
    try:
        if os.path.exists(tokens_path):
            with open(tokens_path, 'r', encoding='utf-8') as f:
                tokens = [line.strip() for line in f.read().split('\n') if line.strip()]
                return tokens if tokens else []
        return []
    except Exception as e:
        print(f"加载tokens失败: {e}")
        return []

def parse_api_tokens(token_input):
    """解析输入的API Tokens（支持逗号、分号、换行分隔）"""
    if not token_input or token_input.strip() in ["", "***已保存***"]:
        return load_api_tokens()
    
    # 支持多种分隔符
    import re
    tokens = re.split(r'[,;\n]+', token_input)
    return [token.strip() for token in tokens if token.strip()]

def tensor_to_base64_url(image_tensor):
    try:
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor.squeeze(0)
        
        if image_tensor.max() <= 1.0:
            image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        else:
            image_np = image_tensor.cpu().numpy().astype(np.uint8)
        
        pil_image = Image.fromarray(image_np)
        
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/jpeg;base64,{img_base64}"
        
    except Exception as e:
        print(f"图像转换失败: {e}")
        raise Exception(f"图像格式转换失败: {str(e)}")


class ModelScopeImageNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        config = load_config()
        saved_tokens = load_api_tokens()
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": config.get("default_prompt", "A beautiful landscape")
                }),
                "api_tokens": ("STRING", {
                    "default": "***已保存{}个Token***".format(len(saved_tokens)) if saved_tokens else "",
                    "placeholder": "请输入API Token（支持多个，用逗号/换行分隔）" if not saved_tokens else "留空使用已保存的Token",
                    "multiline": True
                }),
            },
            "optional": {
                "model": (config.get("image_models", ["Qwen/Qwen-Image"]), {
                    "default": config.get("default_model", "Qwen/Qwen-Image")
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": config.get("default_negative_prompt", "")
                }),
                "width": ("INT", {
                    "default": config.get("default_width", 512),
                    "min": 64,
                    "max": 2048,
                    "step": 64
                }),
                "height": ("INT", {
                    "default": config.get("default_height", 512),
                    "min": 64,
                    "max": 2048,
                    "step": 64
                }),
                "seed": ("INT", {
                    "default": config.get("default_seed", -1),
                    "min": -1,
                    "max": 2147483647
                }),
                "steps": ("INT", {
                    "default": config.get("default_steps", 30),
                    "min": 1,
                    "max": 100
                }),
                "guidance": ("FLOAT", {
                    "default": config.get("default_guidance", 7.5),
                    "min": 1.5,
                    "max": 20.0,
                    "step": 0.1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "ModelScopeAPI"
    
    def generate_image(self, prompt, api_tokens, model="Qwen/Qwen-Image", negative_prompt="", width=512, height=512, seed=-1, steps=30, guidance=7.5):
        config = load_config()
        tokens = parse_api_tokens(api_tokens)
        
        if not tokens:
            raise Exception("请提供至少一个有效的API Token")
        
        # 保存Token（如果提供了新的）
        if api_tokens and api_tokens.strip() not in ["", "***已保存{}个Token***".format(len(load_api_tokens()))]:
            if save_api_tokens(tokens):
                print(f"✅ 已保存 {len(tokens)} 个API Token")
            else:
                print("⚠️ API Token保存失败，但不影响当前使用")
        
        # 轮询尝试每个Token
        last_exception = None
        for i, token in enumerate(tokens):
            try:
                print(f"🔄 尝试使用第 {i+1} 个API Token...")
                url = 'https://api-inference.modelscope.cn/v1/images/generations'
                payload = {
                    'model': model,
                    'prompt': prompt,
                    'size': f"{width}x{height}",
                    'steps': steps,
                    'guidance': guidance
                }
                if negative_prompt.strip():
                    payload['negative_prompt'] = negative_prompt
                if seed != -1:
                    payload['seed'] = seed
                else:
                    import random
                    random_seed = random.randint(0, 2147483647)
                    payload['seed'] = random_seed
                
                headers = {
                    'Authorization': f'Bearer {token}',
                    'Content-Type': 'application/json',
                    'X-ModelScope-Async-Mode': 'true'
                }
                
                submission_response = requests.post(
                    url, 
                    data=json.dumps(payload, ensure_ascii=False).encode('utf-8'), 
                    headers=headers,
                    timeout=config.get("timeout", 60)
                )
                
                if submission_response.status_code == 400:
                    # 尝试使用最小参数重试
                    minimal_payload = {
                        'model': model,
                        'prompt': prompt
                    }
                    submission_response = requests.post(
                        url,
                        data=json.dumps(minimal_payload, ensure_ascii=False).encode('utf-8'),
                        headers=headers,
                        timeout=config.get("timeout", 60)
                    )
                
                if submission_response.status_code != 200:
                    raise Exception(f"API请求失败: {submission_response.status_code}, {submission_response.text}")
                
                submission_json = submission_response.json()
                image_url = None
                if 'task_id' in submission_json:
                    task_id = submission_json['task_id']
                    print(f"🕒 已提交任务，任务ID: {task_id}，开始轮询...")
                    poll_start = time.time()
                    max_wait_seconds = max(60, config.get('timeout', 720))
                    while True:
                        task_resp = requests.get(
                            f"https://api-inference.modelscope.cn/v1/tasks/{task_id}",
                            headers={
                                'Authorization': f'Bearer {token}',
                                'X-ModelScope-Task-Type': 'image_generation'
                            },
                            timeout=config.get("image_download_timeout", 120)
                        )
                        if task_resp.status_code != 200:
                            raise Exception(f"任务查询失败: {task_resp.status_code}, {task_resp.text}")
                        task_data = task_resp.json()
                        status = task_data.get('task_status')
                        if status == 'SUCCEED':
                            output_images = task_data.get('output_images') or []
                            if not output_images:
                                raise Exception("任务成功但未返回图片URL")
                            image_url = output_images[0]
                            print("✅ 任务完成，开始下载图片...")
                            break
                        if status == 'FAILED':
                            raise Exception(f"任务失败: {task_data}")
                        if time.time() - poll_start > max_wait_seconds:
                            raise Exception("任务轮询超时，请稍后重试或降低并发")
                        time.sleep(5)
                elif 'images' in submission_json and len(submission_json['images']) > 0:
                    image_url = submission_json['images'][0]['url']
                    print(f"⬇️ 下载生成的图片...")
                else:
                    raise Exception(f"未识别的API返回格式: {submission_json}")
                
                img_response = requests.get(image_url, timeout=config.get("image_download_timeout", 30))
                if img_response.status_code != 200:
                    raise Exception(f"图片下载失败: {img_response.status_code}")
                
                pil_image = Image.open(BytesIO(img_response.content))
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                image_np = np.array(pil_image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)[None,]
                print(f"🎉 图片处理完成！使用的第 {i+1} 个API Token")
                return (image_tensor,)
                
            except Exception as e:
                last_exception = e
                print(f"⚠️ 第 {i+1} 个API Token失败: {str(e)}")
                if i < len(tokens) - 1:  # 不是最后一个Token
                    print(f"➡️ 尝试下一个API Token...")
                    continue
                else:
                    break  # 所有Token都失败了
        
        # 所有Token都失败
        raise Exception(f"所有 {len(tokens)} 个API Token都失败了。最后的错误: {str(last_exception)}")


class ModelScopeImageEditNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        config = load_config()
        saved_tokens = load_api_tokens()
        
        # 获取模型列表
        edit_models = config.get("image_edit_models", ["Qwen/Qwen-Image-Edit"])
        gen_models = config.get("image_models", ["Qwen/Qwen-Image"])

        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "修改图片中的内容"
                }),
                "api_tokens": ("STRING", {
                    "default": "***已保存{}个Token***".format(len(saved_tokens)) if saved_tokens else "",
                    "placeholder": "请输入API Token（支持多个，用逗号/换行分隔）" if not saved_tokens else "留空使用已保存的Token",
                    "multiline": True
                }),
                "image_gen_mode": ("BOOLEAN", {
                    "default": False,
                    "label_on": "图生图模式",
                    "label_off": "图像编辑模式"
                }),
            },
            "optional": {
                "gen_model": (gen_models, {
                    "default": gen_models[0] if gen_models else "Qwen/Qwen-Image"
                }),
                "edit_model": (edit_models, {
                    "default": edit_models[0] if edit_models else "Qwen/Qwen-Image-Edit"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 1664,
                    "step": 8
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 1664,
                    "step": 8
                }),
                "steps": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "guidance": ("FLOAT", {
                    "default": 3.5,
                    "min": 1.5,
                    "max": 20.0,
                    "step": 0.1
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_image",)
    FUNCTION = "edit_image"
    CATEGORY = "ModelScopeAPI"

    def edit_image(self, image, prompt, api_tokens, image_gen_mode=False, gen_model="Qwen/Qwen-Image", 
                   edit_model="Qwen/Qwen-Image-Edit", negative_prompt="", 
                   width=512, height=512, steps=30, guidance=3.5, seed=-1):
        config = load_config()
        tokens = parse_api_tokens(api_tokens)
        
        if not tokens:
            raise Exception("请提供至少一个有效的API Token")
        
        # 保存Token（如果提供了新的）
        if api_tokens and api_tokens.strip() not in ["", "***已保存{}个Token***".format(len(load_api_tokens()))]:
            if save_api_tokens(tokens):
                print(f"✅ 已保存 {len(tokens)} 个API Token")
            else:
                print("⚠️ API Token保存失败，但不影响当前使用")
        
        # 根据开关选择使用的模型
        if image_gen_mode:
            model = gen_model
            mode_name = "图生图"
        else:
            model = edit_model
            mode_name = "图像编辑"

        # 轮询尝试每个Token
        last_exception = None
        for i, token in enumerate(tokens):
            try:
                print(f"🔄 尝试使用第 {i+1} 个API Token...")
                
                # 将图像转换为临时文件并上传获取URL
                temp_img_path = None
                image_url = None
                try:
                    # 保存图像到临时文件
                    temp_img_path = os.path.join(tempfile.gettempdir(), f"qwen_edit_temp_{int(time.time())}.jpg")
                    if len(image.shape) == 4:
                        img = image[0]
                    else:
                        img = image
                    
                    i = 255. * img.cpu().numpy()
                    img_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                    img_pil.save(temp_img_path)
                    print(f"✅ 图像已保存到临时文件: {temp_img_path}")
                    
                    # 上传图像到kefan.cn获取URL
                    upload_url = 'https://ai.kefan.cn/api/upload/local'
                    with open(temp_img_path, 'rb') as img_file:
                        files = {'file': img_file}
                        upload_response = requests.post(
                            upload_url,
                            files=files,
                            timeout=30
                        )
                        if upload_response.status_code == 200:
                            upload_data = upload_response.json()
                            if upload_data.get('success') == True and 'data' in upload_data:
                                image_url = upload_data['data']
                                print(f"✅ 图像已上传成功，获取URL: {image_url}")
                            else:
                                print(f"⚠️ 图像上传返回错误: {upload_response.text}")
                        else:
                            print(f"⚠️ 图像上传失败: {upload_response.status_code}, {upload_response.text}")
                except Exception as e:
                    print(f"⚠️ 图像上传异常: {str(e)}")
                
                # 如果上传失败，回退到base64
                if not image_url:
                    print("⚠️ 图像URL获取失败，回退到使用base64")
                    image_data = tensor_to_base64_url(image)
                    payload = {
                        'model': model,
                        'prompt': prompt,
                        'image': image_data
                    }
                else:
                    payload = {
                        'model': model,
                        'prompt': prompt,
                        'image_url': image_url
                    }
                
                if negative_prompt.strip():
                    payload['negative_prompt'] = negative_prompt
                
                # 添加新参数
                if width != 512 or height != 512:
                    size = f"{width}x{height}"
                    payload['size'] = size
                
                if steps != 30:
                    payload['steps'] = steps
                
                if guidance != 3.5:
                    payload['guidance'] = guidance
                
                if seed != -1:
                    payload['seed'] = seed
                
                headers = {
                    'Authorization': f'Bearer {token}',
                    'Content-Type': 'application/json',
                    'X-ModelScope-Async-Mode': 'true'
                }
                
                print(f"🖼️ 开始{mode_name}...")
                print(f"✏️ 编辑提示: {prompt}")
                print(f"🧠 使用模型: {model}")
                
                url = 'https://api-inference.modelscope.cn/v1/images/generations'
                submission_response = requests.post(
                    url,
                    data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                    headers=headers,
                    timeout=config.get("timeout", 60)
                )
                
                if submission_response.status_code != 200:
                    raise Exception(f"API请求失败: {submission_response.status_code}, {submission_response.text}")
                
                submission_json = submission_response.json()
                result_image_url = None
                
                if 'task_id' in submission_json:
                    task_id = submission_json['task_id']
                    print(f"🕒 已提交任务，任务ID: {task_id}，开始轮询...")
                    poll_start = time.time()
                    max_wait_seconds = max(60, config.get('timeout', 720))
                    
                    while True:
                        task_resp = requests.get(
                            f"https://api-inference.modelscope.cn/v1/tasks/{task_id}",
                            headers={
                                'Authorization': f'Bearer {token}',
                                'X-ModelScope-Task-Type': 'image_generation'
                            },
                            timeout=config.get("image_download_timeout", 120)
                        )
                        
                        if task_resp.status_code != 200:
                            raise Exception(f"任务查询失败: {task_resp.status_code}, {task_resp.text}")
                        
                        task_data = task_resp.json()
                        status = task_data.get('task_status')
                        
                        if status == 'SUCCEED':
                            output_images = task_data.get('output_images') or []
                            if not output_images:
                                raise Exception("任务成功但未返回图片URL")
                            result_image_url = output_images[0]
                            print("✅ 任务完成，开始下载编辑后的图片...")
                            break
                            
                        if status == 'FAILED':
                            error_message = task_data.get('errors', {}).get('message', '未知错误')
                            error_code = task_data.get('errors', {}).get('code', '未知错误码')
                            raise Exception(f"任务失败: 错误码 {error_code}, 错误信息: {error_message}")
                            
                        if time.time() - poll_start > max_wait_seconds:
                            raise Exception("任务轮询超时，请稍后重试或降低并发")
                            
                        time.sleep(5)
                else:
                    raise Exception(f"未识别的API返回格式: {submission_json}")
                
                img_response = requests.get(result_image_url, timeout=config.get("image_download_timeout", 30))
                if img_response.status_code != 200:
                    raise Exception(f"图片下载失败: {img_response.status_code}")
                
                pil_image = Image.open(BytesIO(img_response.content))
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                image_np = np.array(pil_image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)[None,]
                
                # 清理临时文件
                if temp_img_path and os.path.exists(temp_img_path):
                    try:
                        os.remove(temp_img_path)
                    except:
                        pass
                
                print(f"🎉 {mode_name}完成！使用的第 {i+1} 个API Token")
                return (image_tensor,)
                
            except Exception as e:
                last_exception = e
                print(f"⚠️ 第 {i+1} 个API Token失败: {str(e)}")
                # 清理临时文件
                if temp_img_path and os.path.exists(temp_img_path):
                    try:
                        os.remove(temp_img_path)
                    except:
                        pass
                if i < len(tokens) - 1:  # 不是最后一个Token
                    print(f"➡️ 尝试下一个API Token...")
                    continue
                else:
                    break  # 所有Token都失败了
        
        # 所有Token都失败
        raise Exception(f"所有 {len(tokens)} 个API Token都失败了。最后的错误: {str(last_exception)}")


# 节点映射
NODE_CLASS_MAPPINGS = {
    "ModelScopeImageNode": ModelScopeImageNode,
    "ModelScopeImageEditNode": ModelScopeImageEditNode
}
 
NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelScopeImageNode": "ModelScope-Image 生图节点",
    "ModelScopeImageEditNode": "ModelScope-Image 图像编辑节点"
}
