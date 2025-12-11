import requests
import json
import time
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import os
import base64
import tempfile
 
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("⚠️ 警告: 未安装openai库，图生文功能将不可用")
    print("请运行: pip install openai")
    OPENAI_AVAILABLE = False
    OpenAI = None
 
def load_config():
    config_path = os.path.join.join(os.path.dirname(__file__), 'modelscope_config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {
            "default_model": "Qwen/Qwen-Image",
            "timeout": 720,
            "image_download_timeout": 30,
            "default_prompt": "A beautiful landscape",
            "api_token": ""  # 确保默认默认配置中添加api_token字段
        }
 
def save_config(config):
    """保存配置到modelscope_config.json"""
    config_path = os.path.join(os.path.dirname(__file__), 'modelscope_config.json')
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存配置失败: {e}")
        return False
 
def load_api_token():
    """仅从modelscope_config.json读取API Token"""
    try:
        cfg = load_config()
        return cfg.get("api_token", "").strip()
    except Exception as e:
        print(f"读取config.json中的token失败: {e}")
        return ""
 
def save_api_token(token):
    """仅将API Token保存到modelscope_config.json"""
    try:
        cfg = load_config()
        cfg["api_token"] = token.strip()
        return save_config(cfg)
    except Exception as e:
        print(f"保存token失败: {e}")
        return False
 
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
 
class ModelScopeVisionNode:
    def __init__(self):
        pass
 
    @classmethod
    def INPUT_TYPES(cls):
        if not OPENAI_AVAILABLE:
            return {
                "required": {
                    "error_message": ("STRING", {
                        "default": "请先安装openai库: pip install openai",
                        "multiline": True
                    }),
                }
            }
        config = load_config()
        saved_token = load_api_token()
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": config.get("default_prompt", "描述这幅图")
                }),
                "api_token": ("STRING", {
                    "default": "",
                    "placeholder": "请输入您的魔搭API Token",
                    "multiline": False
                }),
            },
            "optional": {
                "model": (config.get("vision_models", ["stepfun-ai/step3"]), {
                    "default": config.get("default_vision_model", "stepfun-ai/step3")
                }),
                "max_tokens": ("INT", {
                    "default": 1000,
                    "min": 100,
                    "max": 4000
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1
                }),
            }
        }
 
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "analyze_image"
    CATEGORY = "ModelScopeAPI"
 
    def analyze_image(self, image=None, prompt="", api_token="", model="stepfun-ai/step3", max_tokens=1000, temperature=0.7, error_message=""):
        if not OPENAI_AVAILABLE:
            return ("请先安装openai库: pip install openai",)
        
        config = load_config()
        
        if not api_token or api_token.strip() == "":
            api_token = load_api_token()
            if not api_token or api_token.strip() == "":
                raise Exception("请输入有效的API Token或确保已保存token")
        
        saved_token = load_api_token()
        if api_token != saved_token:
            if save_api_token(api_token):
                print("✅ API Token已自动保存到modelscope_config.json")
            else:
                print("⚠️ API Token保存失败，但不影响当前使用")
        
        try:
            print(f"🔍 开始分析图像...")
            print(f"📝 提示词: {prompt}")
            print(f"🤖 模型: {model}")
            
            image_url = tensor_to_base64_url(image)
            print(f"🖼️ 图像已转换为base64格式")
            
            client = OpenAI(
                base_url='https://api-inference.modelscope.cn/v1',
                api_key=api_token
            )
            
            messages = [{
                'role': 'user',
                'content': [{
                    'type': 'text',
                    'text': prompt,
                }, {
                    'type': 'image_url',
                    'image_url': {
                        'url': image_url,
                    },
                }],
            }]
            
            print(f"🚀 发送API请求...")
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            
            description = response.choices[0].message.content
            print(f"✅ 分析完成!")
            print(f"📄 结果: {description[:100]}...")
            
            return (description,)
            
        except Exception as e:
            error_msg = f"图像分析失败: {str(e)}"
            print(f"❌ {error_msg}")
            return (error_msg,)
 
if OPENAI_AVAILABLE:
    NODE_CLASS_MAPPINGS = {
        "ModelScopeVisionNode": ModelScopeVisionNode
    }
     
    NODE_DISPLAY_NAME_MAPPINGS = {
        "ModelScopeVisionNode": "ModelScope-Vision 图生文节点"
    }
else:
    class OpenAINotInstalledNode:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "install_command": ("STRING", {
                        "default": "pip install openai",
                        "multiline": False
                    }),
                }
            }
        
        RETURN_TYPES = ("STRING",)
        RETURN_NAMES = ("message",)
        FUNCTION = "show_install_message"
        CATEGORY = "ModelScopeAPI"
        
        def show_install_message(self, install_command):
            return ("请先安装openai库才能使用图生文功能: " + install_command,)
    
    NODE_CLASS_MAPPINGS = {
        "ModelScopeVisionNode": OpenAINotInstalledNode
    }
 
    NODE_DISPLAY_NAME_MAPPINGS = {
        "ModelScopeVisionNode": "ModelScope-Vision 图生文节点 (需要安装openai)"
    }