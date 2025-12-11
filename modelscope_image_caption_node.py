import requests
import json
import time
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import os
import base64
import re
from .modelscope_image_node import load_config, save_config, tensor_to_base64_url

# 检查openai库是否可用
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# 仅与modelscope_config.json交互的API Token管理函数
def load_api_tokens():
    try:
        cfg = load_config()
        tokens_from_cfg = cfg.get("api_tokens", [])
        if tokens_from_cfg and isinstance(tokens_from_cfg, list):
            return [token.strip() for token in tokens_from_cfg if token.strip()]
    except Exception as e:
        print(f"读取config中的tokens失败: {e}")
    return []

def save_api_tokens(tokens):
    try:
        cfg = load_config()
        cfg["api_tokens"] = tokens
        return save_config(cfg)
    except Exception as e:
        print(f"保存tokens到config失败: {e}")
        return False

class ModelScopeImageCaptionNode:
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
        saved_tokens = load_api_tokens()
        # 定义支持的模型列表
        supported_models = [
            "Qwen/Qwen3-VL-8B-Instruct",
            "Qwen/Qwen3-VL-235B-A22B-Instruct"
        ]
        return {
            "required": {
                "api_tokens": ("STRING", {
                    "default": f"***已保存{len(saved_tokens)}个Token***" if saved_tokens else "",
                    "placeholder": "请输入API Token（支持多个，用逗号/换行分隔）",
                    "multiline": True
                }),
            },
            "optional": {
                # 关键修改：将image设置为可选输入
                "image": ("IMAGE", {"optional": True}),
                "prompt1": ("STRING", {
                    "multiline": True,
                    "default": "详细描述这张图片的内容，包括主体、背景、颜色、风格等信息"
                }),
                "prompt2": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "model": (supported_models, {
                    "default": "Qwen/Qwen3-VL-8B-Instruct"
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
    FUNCTION = "generate_caption"
    CATEGORY = "ModelScopeAPI"

    def parse_api_tokens(self, token_input):
        """解析输入的多个API Token（支持逗号、分号、换行分隔）"""
        if not token_input or token_input.strip() in ["", f"***已保存{len(load_api_tokens())}个Token***"]:
            return load_api_tokens()
        
        # 支持多种分隔符拆分Token
        tokens = re.split(r'[,;\n]+', token_input)
        return [token.strip() for token in tokens if token.strip()]
    
    def create_blank_image(self, width=64, height=64):
        """创建空白图像张量（符合ComfyUI的图像格式要求）"""
        # 创建白色背景的RGB图像
        blank_np = np.ones((height, width, 3), dtype=np.uint8) * 255
        # 转换为ComfyUI格式的张量 (batch, height, width, channels)
        blank_tensor = torch.from_numpy(blank_np).unsqueeze(0).float() / 255.0
        return blank_tensor

    def generate_caption(self, image=None, api_tokens="", prompt1="详细描述这张图片的内容", prompt2="", model="Qwen/Qwen3-VL-8B-Instruct", max_tokens=1000, temperature=0.7):
        if not OPENAI_AVAILABLE:
            return ("请先安装openai库: pip install openai",)
        
        # 关键修改：处理输入图像为空的情况
        if image is None:
            print("⚠️ 未输入图像，自动生成空白图像作为输入")
            image = self.create_blank_image()
        
        # 处理提示词合并
        prompt_parts = []
        if prompt1.strip():
            prompt_parts.append(prompt1.strip())
        if prompt2.strip():
            prompt_parts.append(prompt2.strip())
        
        if not prompt_parts:
            prompt = "详细描述这张图片的内容，包括主体、背景、颜色、风格等信息"
        else:
            prompt = ", ".join(prompt_parts)
        
        # 解析Token列表
        tokens = self.parse_api_tokens(api_tokens)
        if not tokens:
            raise Exception("请提供至少一个有效的API Token")
        
        # 保存新Token（如果有变化）
        saved_tokens = load_api_tokens()
        if api_tokens.strip() not in ["", f"***已保存{len(saved_tokens)}个Token***"]:
            if save_api_tokens(tokens):
                print(f"✅ 已保存 {len(tokens)} 个API Token")
            else:
                print("⚠️ API Token保存失败，但不影响当前使用")
        
        try:
            print(f"🔍 开始生成图像描述...")
            print(f"📝 提示词: {prompt}")
            print(f"🤖 模型: {model}")
            print(f"🔑 可用Token数量: {len(tokens)}")
            
            # 转换图像为base64格式
            image_url = tensor_to_base64_url(image)
            print(f"🖼️ 图像已转换为base64格式")
            
            # 构建消息体
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
            
            # 轮询尝试每个Token
            last_exception = None
            for i, token in enumerate(tokens):
                try:
                    print(f"🔄 尝试使用第 {i+1}/{len(tokens)} 个Token...")
                    
                    client = OpenAI(
                        base_url='https://api-inference.modelscope.cn/v1',
                        api_key=token
                    )
                    
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=False
                    )
                    
                    description = response.choices[0].message.content
                    print(f"✅ 第 {i+1} 个Token调用成功!")
                    print(f"📄 结果预览: {description[:100]}...")
                    return (description,)
                    
                except Exception as e:
                    last_exception = e
                    print(f"❌ 第 {i+1} 个Token调用失败: {str(e)}")
                    if i < len(tokens) - 1:
                        print(f"⏳ 准备尝试下一个Token...")
            
            # 所有Token都失败
            raise Exception(f"所有Token均调用失败: {str(last_exception)}")
            
        except Exception as e:
            error_msg = f"图像描述生成失败: {str(e)}"
            print(f"❌ {error_msg}")
            return (error_msg,)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "ModelScopeImageCaptionNode": ModelScopeImageCaptionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelScopeImageCaptionNode": "ModelScope 图像描述生成"
}