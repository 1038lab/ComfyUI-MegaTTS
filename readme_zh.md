# ComfyUI-MegaTTS 

([English](readme.md) / **中文**)

基于字节跳动[MegaTTS3](https://huggingface.co/ByteDance/MegaTTS3)的ComfyUI自定义节点，支持中英双语的高质量文本转语音合成及声音克隆功能。

![Comfyui-MegaTTS-NodeSamples zh](https://github.com/user-attachments/assets/750443fe-d6ab-4340-a758-a85bb10aa07f)

## 更新日志

### 版本 1.0.2
- 重构了代码和自定义节点，以优化性能和更好的GPU资源管理。
- 增强了内存管理功能，以防止低显存用户内存不足。
- 国际化i18n, 支持中文英文

### 版本 1.0.1
- 修复了bug

## 功能特点

- **高质量语音合成**：从文本输入生成自然流畅的语音
- **声音克隆**：只需一个短样本即可克隆任何声音（需要WAV和NPY文件）
- **双语支持**：支持中文和英文文本，具有代码切换能力
- **高级参数控制**：精细调节生成质量、发音准确度和声音相似度
- **内存管理**：内置功能优化GPU资源使用
- **自动模型下载**：需要时自动下载模型

## 安装

### 前提条件

- 已安装并正常运行的ComfyUI
- 推荐Python 3.10+
- 支持CUDA的GPU，至少4GB显存（推荐8GB+以获得更高质量）

### 安装步骤

1. 将此仓库克隆到ComfyUI的`custom_nodes`目录：
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/1038lab/ComfyUI-MegaTTS.git
   ```

2. 安装所需依赖：
   ```bash
   cd ComfyUI-MegaTTS
   pip install -r requirements.txt
   ```

3. 节点将在首次使用时自动下载所需模型，或者您可以手动下载它们：

## 模型与手动下载

此扩展使用经过修改的字节跳动MegaTTS3模型。虽然模型会在首次使用时自动下载，但您也可以从Hugging Face手动下载它们：

### 模型结构

模型按以下结构组织：
```
model_path/TTS/MegaTTS3/
  ├── diffusion_transformer/
  │   ├── config.yaml
  │   └── model_only_last.ckpt
  ├── wavvae/
  │   ├── config.yaml
  │   └── decoder.ckpt
  ├── duration_lm/
  │   ├── config.yaml
  │   └── model_only_last.ckpt
  ├── aligner_lm/
  │   ├── config.yaml
  │   └── model_only_last.ckpt
  └── g2p/
      ├── config.json
      ├── model.safetensors
      ├── generation_config.json
      ├── tokenizer_config.json
      ├── special_tokens_map.json
      ├── tokenizer.json
      ├── vocab.json
      └── merges.txt
```

### 手动下载选项

1. **从Hugging Face直接下载**：
   - 访问[ByteDance/MegaTTS3仓库](https://huggingface.co/ByteDance/MegaTTS3/tree/main)
   - 从仓库下载每个子文件夹：
     - [aligner_lm](https://huggingface.co/ByteDance/MegaTTS3/tree/main/aligner_lm)
     - [diffusion_transformer](https://huggingface.co/ByteDance/MegaTTS3/tree/main/diffusion_transformer)
     - [duration_lm](https://huggingface.co/ByteDance/MegaTTS3/tree/main/duration_lm)
     - [g2p](https://huggingface.co/ByteDance/MegaTTS3/tree/main/g2p)
     - [wavvae](https://huggingface.co/ByteDance/MegaTTS3/tree/main/wavvae)
   - 将下载的文件放在`comfyui/models/TTS/MegaTTS3/`下的相应目录中

2. **使用Hugging Face CLI**：
   ```bash
   # 如果您还没有huggingface_hub，请先安装
   pip install huggingface_hub
   
   # 下载所有模型
   python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ByteDance/MegaTTS3', local_dir='comfyui/models/TTS/MegaTTS3/')"
   ```

## 声音文件夹与声音制作器

![Voice_Maker](https://github.com/user-attachments/assets/b3713f9a-2f70-4bf0-a6f4-7e3110bba987)

> [!重要]  
> WaveVAE编码器目前不可用。
>
> 出于安全考虑，字节跳动未上传WaveVAE编码器。
>
> 您只能使用预先提取的潜在变量（.npy文件）进行推理。
>
> 要为特定说话者合成语音，请确保相应的WAV和NPY文件位于同一目录中。
>
> 有关获取必要文件或提交语音样本的详细信息，请参阅[字节跳动MegaTTS3仓库](https://github.com/bytedance/MegaTTS3)。

### 声音文件夹结构

扩展需要一个`Voices`文件夹来存储参考声音样本及其提取的特征：

```
Voices/
├── sample1.wav     # 参考音频文件
├── sample1.npy     # 从音频文件提取的特征
├── sample2.wav     # 另一个参考音频
└── sample2.npy     # 对应的特征
```

### 获取声音样本和NPY文件

1. **下载预提取样本**：
   - 样本声音WAV和NPY文件可在此Google Drive文件夹中找到：[声音样本和NPY文件](https://drive.google.com/drive/folders/1QhcHWcy20JfqWjgqZX1YM3I6i9u4oNlr?usp=sharing)
   - 此文件夹包含按子文件夹组织的预提取NPY文件及其对应的WAV样本

2. **提交您自己的声音样本**：
   - 如果您想使用自己的声音，可以将样本提交到此Google Drive文件夹：[声音提交队列](https://drive.google.com/drive/folders/1gCWL1y_2xu9nIFhUX_OW5MbcFuB7J5Cl?usp=sharing)
   - 您的样本应该是清晰的音频，背景噪音最小，时长在24秒以内
   - 经过安全验证后，字节跳动团队将提取并提供您样本的NPY文件

3. **使用声音制作器生成NPY文件**：
   - 使用声音制作器节点自动处理您的音频并生成NPY文件
   - 虽然此方法方便，但质量可能不如官方提取的NPY文件
   - 最适合使用您自己的声音样本进行快速测试和实验

### 声音制作器节点

此扩展包含一个**声音制作器**自定义节点，帮助您准备声音样本：

- **声音制作器节点功能**：
  - 将任何音频文件转换为所需的24kHz WAV格式
  - 从WAV样本中提取NPY特征文件
  - 处理和优化声音样本以获得更好的质量
  - 自动将处理后的文件保存到Voices文件夹

**如何使用声音制作器**：
1. 从🧪AILab/🔊Audio类别添加"Voice Maker"节点
2. 连接音频输入或从您的计算机选择文件
3. 配置处理选项（标准化、修剪等）
4. 运行节点以生成带NPY文件的即用声音样本

### 关于WAV和NPY文件

- **WAV文件**：这些是您想要克隆的实际声音样本（推荐24kHz）
- **NPY文件**：这些包含声音克隆所必需的提取特征

### 声音格式要求

为获得最佳效果：
- **采样率**：24kHz（如果不同会自动转换）
- **音频格式**：推荐WAV，但也支持MP3、M4A和其他格式
- **时长**：5-24秒的清晰语音
- **质量**：干净的录音，背景噪音最小

## 参数调整

### 控制声音口音

该模型提供对口音和发音的出色控制：

- **保留说话者口音**：
  - 将发音强度(p_w)设置为较低值(1.0-1.5)
  - 这对于想要保留口音的跨语言TTS很有用

- **标准发音**：
  - 将发音强度(p_w)设置为较高值(2.5-4.0)
  - 这有助于产生更标准的发音，无论源口音如何

- **情感或表现力丰富的语音**：
  - 增加声音相似度(t_w)参数(2.0-5.0)
  - 保持发音强度(p_w)在中等水平(1.5-2.5)

### 推荐参数组合

| 用例 | p_w (发音强度) | t_w (声音相似度) |
|----------|------------------------------|------------------------|
| 标准TTS | 2.0 | 3.0 |
| 保留口音 | 1.0-1.5 | 3.0-5.0 |
| 跨语言(标准) | 3.0-4.0 | 3.0-5.0 |
| 情感语音 | 1.5-2.5 | 3.0-5.0 |
| 噪音参考音频 | 3.0-5.0 | 3.0-5.0 |

## 节点

此扩展提供三个主要节点：

### 1. Mega TTS (高级)

具有完整参数控制的全功能TTS节点。

**输入：**
- `input_text` - 要转换为语音的文本
- `language` - 语言选择(en：英语，zh：中文)
- `generation_quality` - 控制扩散步数(越高=质量越好但速度越慢)
- `pronunciation_strength` (p_w) - 控制发音准确度(值越高产生更标准的发音)
- `voice_similarity` (t_w) - 控制与参考声音的相似度(值越高产生更接近参考的语音)
- `reference_voice` - Voices文件夹中的参考声音文件

**输出：**
- `AUDIO` - WAV格式的生成音频
- `LATENT` - 用于进一步处理的音频潜在表示

### 2. Mega TTS (简单)

带默认参数的简化TTS节点，便于快速使用。

**输入：**
- `input_text` - 要转换为语音的文本
- `language` - 语言选择(en：英语，zh：中文)
- `reference_voice` - Voices文件夹中的参考声音文件

**输出：**
- `AUDIO` - WAV格式的生成音频

### 3. Mega TTS (清理内存)

TTS处理后释放GPU内存的实用节点。

## 参数说明

| 参数 | 说明 | 推荐值 |
|-----------|-------------|-------------------|
| **generation_quality** | 控制扩散步数。值越高，质量越好，但生成时间越长。 | 默认：10。范围：1-50。快速测试：1-5，最终输出：15-30。 |
| **pronunciation_strength** (p_w) | 控制输出如何紧密遵循标准发音。 | 默认：2.0。范围：1.0-5.0。保留口音：1.0-1.5，标准发音：2.5-4.0。 |
| **voice_similarity** (t_w) | 控制输出与参考声音的相似度。 | 默认：3.0。范围：1.0-5.0。更具表现力且保留声音特征的输出：3.0-5.0。 |

## 声音克隆

### 添加参考声音

1. 将您的声音WAV文件放在`Voices`文件夹中
2. 每个声音需要两个文件：
   - `voice_name.wav` - 声音样本文件（推荐24kHz采样率，5-10秒清晰语音）
   - `voice_name.npy` - 对应的声音特征文件（如果启用声音提取，将自动生成）

### 如何克隆声音

1. 将您的样本WAV文件添加到`Voices`文件夹
2. 首次选择声音时，系统将提取特征文件并保存
3. 在节点的"reference_voice"下拉菜单中选择您的声音
4. 调整"voice_similarity"参数控制声音克隆的强度：
   - 较低值(1.0-2.0)：更自然但与参考声音相似度较低
   - 较高值(3.0-5.0)：与参考声音更相似但可能不太自然

## 高级用法

### 跨语言声音克隆

对于跨语言克隆声音（例如，让英语说话者说中文）：

1. 使用原始语言的干净声音样本
2. 将语言设置为目标语言（例如，中文为"zh"）
3. 增加发音强度(p_w)参数(3.0-4.0)以获得更标准的发音
4. 设置较高的声音相似度(t_w)参数(3.0-5.0)以保持声音特征

### 处理口音

- 保留口音：较低的发音强度(p_w)值(1.0-1.5)
- 标准发音：较高的发音强度(p_w)值(2.5-4.0)

## 致谢

- 原始MegaTTS3模型由[字节跳动](https://github.com/bytedance/MegaTTS3)开发
- MegaTTS3 Hugging Face模型：[ByteDance/MegaTTS3](https://huggingface.co/ByteDance/MegaTTS3)

## 许可证
GPL-3.0 许可证

## 参考文献

- [字节跳动MegaTTS3 GitHub仓库](https://github.com/bytedance/MegaTTS3)
- [字节跳动MegaTTS3 Hugging Face模型](https://huggingface.co/ByteDance/MegaTTS3)
- 原始论文：
  - ["Sparse Alignment Enhanced Latent Diffusion Transformer for Zero-Shot Speech Synthesis"](https://arxiv.org/abs/2502.18924)
  - ["Wavtokenizer: an efficient acoustic discrete codec tokenizer for audio language modeling"](https://arxiv.org/abs/2408.16532) 
