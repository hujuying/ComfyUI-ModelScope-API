# ComfyUI-ModelScope-API
 
ComfyUI-ModelScope-API 是一个强大的 ComfyUI 自定义节点，它架起了 ComfyUI 可视化工作流环境与 ModelScope 丰富模型库之间的桥梁。
 
This is a powerful custom node for ComfyUI that bridges the gap between ComfyUI's visual workflow environment and ModelScope's extensive collection.

---
# 没有Comfyui有需要WebUI界面的可以到龙大的项目
ModelScope API WebUI
基于ModelScope API的现代化Web应用，提供完整的AI模型服务访问界面。

---
## 更新日志：

**2025-12-12 更新**  对图片解析节点增加随机种子选择，现在可以每次生成不同的描述了。

**2025-12-12 更新**  修正ModelScope-Vision 图生文节点报错，修正Qwen-VL模型 ModelScope 图像描述生成节点因拼写错误造成的解析错误（致谢ShmilyStar反馈）

**2025-12-11 更新**  生图模型支持Z-image模型。参考了https://github.com/otsluo/comfyui-modelscope-api 的lora代码对图像生成节点与图像编辑节点增加了lora功能，图像描述节点节点可不输入图片直接使用Qwen3-VL作为大语言模型使用。

**2025-10-23 更新**  ComfyUI-ModelScope-API  图像描述节点增加一个次要提示词输入，可同时输入2条提示词（自动合并）

**2025-10-22 更新**  ComfyUI-ModelScope-API  增加图像描述节点 支持Qwen3-VL系列模型反推

## 🏗️ 核心架构概述
 
该架构的核心是 `ModelScopeAPI` 类，它作为 ComfyUI 与 ModelScope 云服务之间的主要接口。该架构采用模块化设计模式，职责划分清晰，处理从用户输入验证到图像处理和 API 通信的所有事务。
 
### 架构特点
 
- **模块化设计**：清晰的职责分离，易于维护和扩展
- **统一接口**：通过单一 API 类处理所有模型调用
- **云端处理**：无需本地下载模型，直接在云端生成图像
- **参数验证**：完整的输入参数验证和错误处理
 
---
 
### ✨ 功能特性
 
- **多模型支持**：自由输入模型名称，支持 ModelScope 下的所有兼容模型
- **双模式生成**：
  - **文生图模式**：仅通过文本提示词生成图像
  - **图生图模式**：基于输入图像和文本提示词进行图像转换
- **直接的 API 调用**：无需在本地下载模型，直接通过 API 在云端生成图像
- **完整的参数控制**：支持调整分辨率（宽度和高度）、随机种子（Seed）、采样步数（Steps）和提示词引导系数（Guidance）
- **内置图床服务**：自动将输入的图像上传到免费图床，以获取可供 API 使用的公开 URL
- **灵活的模型选择**：在UI界面中可以自由输入任何 ModelScope 支持的模型名称
 
---
 
## 🖼️ 使用示例
 
### 文生图模式
 
[文本提示词] → [ModelScope API Node] → [生成图像]
 
### 图生图模式
 
[输入图像] + [文本提示词] → [ModelScope API Node] → [转换后图像]
 
---
 
## ⚙️ 安装
 
### 方法一：使用 Git
 
1. 打开一个终端或命令行窗口
2. 导航到你的 ComfyUI 安装目录下的 `custom_nodes` 文件夹
   ```bash
   cd /path/to/your/ComfyUI/custom_nodes/
3.运行以下命令克隆本仓库   
git clone https://github.com/hujuying/ComfyUI-ModelScope-API.git
### 方法二：手动下载
点击本页面右上角的 Code 按钮，然后选择 Download ZIP
解压下载的 ZIP 文件
将解压后的文件夹（确保文件夹名为 ComfyUI-ModelScope-API）移动到 ComfyUI 的 custom_nodes 目录下
重启 ComfyUI

### 🚀 使用方法
### 通用步骤
在 ComfyUI 中，通过右键菜单或双击搜索 ModelScope Universal API 来添加此节点
在 api_key 字段中，填入你的 ModelScope API Key
在 model_name 字段中，输入要使用的模型名称（例如：MusePublic/FLUX.1-Kontext-Dev）
在 prompt 字段中，输入你想要的图像描述或修改提示
根据需要调整其他参数
将节点的 IMAGE 输出连接到 PreviewImage 或 SaveImage 节点以查看结果

### 文生图模式

不连接任何图像到节点的 image 输入
仅提供文本提示词，节点将自动进行文生图操作

### 图生图模式

将一个图像输出连接到本节点的 image 输入
提供文本提示词描述你想要对图像进行的修改
节点将基于输入图像进行图生图操作

### 📋 参数说明

参数	类型	范围	用途
api_key	字符串	-	ModelScope API 访问的身份验证
model_name	字符串	-	目标模型的标识符
prompt	字符串	-	用于生成的文本描述
image	图像 (可选)	-	图生图模式的输入图像
width	整数	64-2048	输出图像宽度（像素）
height	整数	64-2048	输出图像高度（像素）
seed	整数	0-2147483647	用于可重现结果的随机种子
steps	整数	1-100	采样步数
guidance	浮点数	1.5-20.0	提示词引导系数

### 🎯 支持的模型

### 官方支持的模型
模型名称	描述	用例
MusePublic/FLUX.1-Kontext-Dev	用于通用图像生成的默认 FLUX 模型	文本生成图像、图像生成图像
MusePublic/Qwen-image	专用于详细场景	复杂构图和细节
MusePublic/Qwen-Image-Edit	图像编辑专用模型	图像修改和增强
MusePublic/489_ckpt_FLUX_1	FLUX 系列变体	高质量图像生成
MAILAND/majicflus_v1	麦橘超然模型	艺术风格生成
以及 ModelScope 平台上其他兼容的模型。

### 💡 最佳实践
提示词增强
在 ModelScope API 之前使用提示词增强节点来改进您的文本描述
后处理
将输出连接到图像增强或放大节点进行最终精修
批量处理
创建多个具有不同种子的 ModelScope API 节点，以生成相同概念的变体
图生图提示词技巧
对于图像到图像生成，提示词应该描述您想要进行的更改，而不是整个图像。例如，如果您输入一张猫的照片，像"让它看起来像水彩画"这样的提示词会比"水彩风格的猫"更有效。

### 🙏 致谢
API服务提供方: 魔搭 ModelScope
模型提供方: MusePublic
图片上传服务: freeimage.host
📄 许可证
本项目采用 MIT License 开源。


这个新的 README.md 内容基于概述页面的架构信息重新组织，增加了核心架构概述部分，优化了参数说明表格，并添加了最佳实践部分，使文档更加完整和实用。

[通用API架构](4-universal-api-architecture)
[快速开始](2-quick-start)
[在ComfyUI中使用ModelScope模型](3-working-with-modelscope-models-in-comfyui)
