# LongCat-Flash-Omni 深度技术分析

> 美团 LongCat 团队开发的全能型 AI 模型 - 560B 参数 (27B 激活) 的多模态大模型完整技术分析

## 模型概述

| 项目 | 规格 |
|------|------|
| **总参数量** | 560B (5600 亿) |
| **激活参数** | 27B (~4.8%) |
| **架构** | Shortcut-connected MoE (Mixture-of-Experts) |
| **模态支持** | 文本、音频、图像、视频 |
| **上下文长度** | 128K tokens |
| **音频帧粒度** | 80ms |
| **推理精度** | FP8 (单节点) / BF16 (多节点) |

## 目录

- [模型架构](#模型架构)
- [推理流程](#推理流程)
- [Token 处理机制](#token-处理机制)
- [Attention 机制](#attention-机制)
- [并行策略](#并行策略)
- [硬件要求](#硬件要求)

---

## 模型架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                     LongCat-Flash-Omni 架构                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │
│    │  Visual      │    │   Audio      │    │    Text      │         │
│    │  Encoder     │    │   Encoder    │    │  Embedding   │         │
│    │  (Univitar)  │    │   (DFSMN)    │    │  (131K vocab)│         │
│    └──────┬───────┘    └──────┬───────┘    └──────┬───────┘         │
│           │                   │                   │                  │
│           └───────────────────┼───────────────────┘                  │
│                               ▼                                      │
│                    ┌──────────────────┐                               │
│                    │  Embedding Fusion│  ← 统一嵌入空间 (7168维)      │
│                    └────────┬─────────┘                               │
│                             │                                         │
│                             ▼                                         │
│                    ┌──────────────────┐                               │
│                    │   LongCat Flash  │  ← MoE 主干 (560B参数)        │
│                    │   (MoE Backbone) │     激活 27B                 │
│                    └────────┬─────────┘                               │
│                             │                                         │
│                             ▼                                         │
│                    ┌──────────────────┐                               │
│                    │  Output Heads     │                               │
│                    │  (Text + Audio)   │                               │
│                    └──────────────────┘                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 视觉编码器 (Univitar)

**配置参数**:
```python
LongCatVisionConfig:
├── num_hidden_layers: 24          # Transformer 层数
├── num_attention_heads: 16        # 注意力头数
├── hidden_size: 1024              # 隐藏层维度
├── intermediate_size: 4224        # FFN 中间层维度
├── patch_size: 14                 # 空间 patch 大小
├── temporal_patch_size: 2         # 时间 patch 大小 (视频)
├── image_size: 1792               # 输入图像分辨率
└── attention_type: "flash_attention"
```

**处理流程**:
```
输入图像 (H×W×3)
    ↓
3D Convolution (kernel: [2, 14, 14])
    ↓
Patch Embeddings (1024维)
    ↓
24× Transformer Layers
    ├── FlashAttention (双向)
    ├── 2D Rotary Position Embedding
    ├── SwiGLU Activation
    └── RMSNorm
    ↓
Vision Projector (1024 → 7168)
    ↓
输出 (7168维, 与文本对齐)
```

### 音频编码器 (DFSMN)

**配置参数**:
```python
LongCatAudioConfig:
├── input_size: 1200      # fbank 特征维度
├── hidden_size: 6144     # FSMN 隐藏层
├── proj_size: 1536       # 投影层维度
├── nlayer: 22            # DFSMN 层数
├── ndnn: 2               # DNN 层数
├── left_order: 10        # 左记忆窗口 (800ms)
├── right_order: 1        # 右记忆窗口 (80ms)
└── activation: relu6
```

**DFSMN 架构**:
```
输入音频特征 (1200维 fbank)
    ↓
22× DFSMN Layers
    ├── Memory Block
    │   └── Depthwise 1D Conv (kernel_size = 12)
    └── FFN Block
        ├── LayerNorm
        ├── Linear (1200 → 6144)
        ├── ReLU6
        └── Linear (6144 → 1200)
    ↓
2× DNN Layers
    ↓
Audio Projector (1200 → 7168)
    ↓
输出 (7168维)
```

**音频帧粒度**: 每帧 **80ms** (`AUDIO_INPUT_FRAME_SEC = 0.08`)

---

## 推理流程

### 完整调用链

```
longcat_omni_demo.py:main()
    │
    ├─→ init_global_config(args)
    │   └─→ set_global_variables(config)
    │
    ├─→ LoncatOmniInfer.__init__(args)
    │   │
    │   ├─→ build_modality_models()
    │   │   ├─→ TextEmbedding()
    │   │   ├─→ LongCatOmniVisionAdaptor()
    │   │   ├─→ LongCatOmniAudioAdaptor()
    │   │   ├─→ AudioEmbedding(audio_head_num=4)
    │   │   ├─→ DataProcessor()
    │   │   └─→ OmniUnifiedPostProcessor()
    │   │
    │   └─→ create_sglang_engine()
    │
    └─→ infer_engine.generate(input, sampling_params)
```

### 单次推理详细流程

```python
# 步骤 1: 数据预处理
def _process_input(input_dict):
    data = self._input_processor.process(input_dict)
    # 返回: prompts, audios, audio_masks, images, grid_shapes

# 步骤 2: 嵌入生成
def _get_input_embedding(input_ids, codecs, audios, images):
    # 2.1 基础文本嵌入
    merged = self.text_embedding(input_ids)

    # 2.2 音频 codec 嵌入 (4个 codebook 相加)
    if codecs is not None:
        audio_embs = self.audio_embedding(codecs)
        for i in range(4):
            merged += audio_embs[i]

    # 2.3 连续音频嵌入 (替换 pad 位置)
    if audios is not None:
        audio_emb = self.audio_adaptor_model(audios, audio_masks)
        merged[audio_pad_mask] = audio_emb

    # 2.4 视觉嵌入 (替换 pad 位置)
    if images is not None:
        vision_emb = self.vision_adaptor_model(images, grid_shapes)
        merged[vision_pad_mask] = vision_emb

    return merged

# 步骤 3: SGLang 推理
async def generate():
    output = await self.sglang_engine.async_generate(
        input_embeds=input_embedding,
        sampling_params={"temperature": 1.0, "max_new_tokens": 4096}
    )
    return output

# 步骤 4: 后处理
def post_processor.process(output):
    text = tokenizer.decode(output["output_ids"])
    waveform = codec_decoder.decode(output["aux_info"]["audio_codes"])
    return ProcessedOutput(text=text, audio_waveform=waveform)
```

### 数据流转图

```
输入 (用户消息)
    ↓
DataProcessor.process()
    ├─→ read_vison_and_audio()
    ├─→ _precess_global_vision_info()
    ├─→ apply_chat_template()
    └─→ __handle_continuity_audio/vision()
    ↓
_get_input_embedding()
    ├─→ text_embedding()
    ├─→ audio_embedding() (相加)
    ├─→ audio_adaptor() (替换)
    └─→ vision_adaptor() (替换)
    ↓
SGLang Engine (MoE Backbone)
    ↓
OmniUnifiedPostProcessor.process()
    ├─→ process_text_output()
    └─→ process_audio_output()
    ↓
最终输出 (text + audio)
```

---

## Token 处理机制

### 特殊 Token 定义

```python
# 音频相关
AUDIO_BOS_TOKEN = "<|audio|>"        # 音频开始
AUDIO_EOS_TOKEN = "<|/audio|>"       # 音频结束
AUDIO_PAD_TOKEN = "<|audio_pad|>"    # 音频填充占位符

# 视觉相关
IMAGE_PAD_TOKEN = "<|image_pad|>"    # 图像填充
DEFAULT_IMAGE_TOKEN = "<image>"

# 对话角色
SYSTEM_BOS_TOKEN = "<begin-of-system>"
USER_BOS_TOKEN = "<begin-of-user>"
ASSISTANT_BOS_TOKEN = "<begin-of-assistant>"

# 音频 Codec
CODEC_EOS_ID = 2
CODEC_PAD_ID = 3
NUM_CODEC_PLACEHOLDERS = 32
```

### Mask Token 映射机制

```python
# 词汇表中预留 mask token
MASK_START_IDX = 110  # <mask_110>, <mask_111>, ...

SPECIAL_TOKEN_TO_MASK = {
    "<|audio|>": "<mask_110>",
    "<|/audio|>": "<mask_111>",
    "<|audio_pad|>": "<mask_112>",
    "</pause>": "<mask_113>",
    "<begin-of-user>": "<mask_115>",
    "<begin-of-assistant>": "<mask_116>",
    "<|image_pad|>": "<mask_118>",
    "<image>": "<mask_119>",
}
```

### 嵌入融合策略

```python
# 文本: 直接 embedding
merged = text_embedding(input_ids)

# 音频 Codec: 相加融合
for i in range(4):
    merged += audio_embedding[i](codecs[:, :, i])

# 连续音频: 替换 pad 位置
merged[audio_pad_mask] = audio_adaptor_embedding

# 视觉: 替换 pad 位置
merged[vision_pad_mask] = vision_adaptor_embedding
```

### 音频 Codec 结构

```
4-codebook 编码:
┌─────────────────────────────────────────────┐
│  Codebook 0: 语义 Token (Semantic)         │
│  Codebook 1-3: 声学 Token (Acoustic)       │
│                                             │
│  每帧 80ms → 4 个 token                    │
│  Codec ID 偏移: +32                         │
└─────────────────────────────────────────────┘
```

---

## Attention 机制

### SGLang Attention

```python
# 支持的 Attention 类型
1. MHA (Multi-Head Attention)
2. MLA (Multi-Head Latent Attention)

# FlashAttention 配置
- use_flash_attention: True
- sliding_window_size: -1 (全 attention) 或 >0 (滑动窗口)
- causal: True (生成阶段)
```

### 长上下文处理 (128K Token)

```python
# 分块注意力
def make_local_attention_virtual_batches(
    attn_chunk_size,      # 注意力块大小
    query_start_loc_np,    # 查询起始位置
    seq_lens_np,          # 序列长度
    block_table,          # 块表
    page_size,            # 页面大小 (16)
):
    # 将长序列分割为局部注意力块
    effective_chunk_size = min(attn_chunk_size, max_seq_len)
    effective_chunk_size = (effective_chunk_size // page_size) * page_size
    return chunks
```

### KV Cache 优化

- **FP8 量化**: 内存节省 50%
- **Paged KV Cache**: 分页存储, 动态管理
- **Prefix Cache**: 缓存公共前缀

---

## 并行策略

### TP + EP 并行架构

```
┌────────────────────────────────────────────────────────────┐
│               TP + EP 并行架构                              │
├────────────────────────────────────────────────────────────┤
│                                                            │
│   Node 0                              Node 1              │
│   ┌────┬────┬────┬────┐              ┌────┬────┬────┬────┐│
│   │GPU0│GPU1│GPU2│GPU3│              │GPU4│GPU5│GPU6│GPU7││
│   ├────┼────┼────┼────┤              ├────┼────┼────┼────┤│
│   │ TP │ TP │ TP │ TP │              │ TP │ TP │ TP │ TP ││
│   │ EP │ EP │ EP │ EP │              │ EP │ EP │ EP │ EP ││
│   │ E0 │ E1 │ E2 │ E3 │              │ E4 │ E5 │ E6 │ E7 ││
│   └────┴────┴────┴────┘              └────┴────┴────┴────┘│
│                                                            │
│   TP (Tensor Parallelism): 模型层内切分                    │
│   EP (Expert Parallelism): MoE 专家分布                    │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 推荐配置

| 配置 | GPU | 精度 | 参数 |
|------|-----|------|------|
| 单节点 | 8× H20-141G | FP8 | `--tp-size 8 --ep-size 8` |
| 双节点 | 16× H800-80G | BF16 | `--tp-size 16 --ep-size 16 --nodes 2` |

---

## 硬件要求

### 最低配置
- **GPU**: 单节点 8× H20-141G (141GB VRAM)
- **精度**: FP8
- **并行**: TP=8, EP=8

### 推荐配置
- **GPU**: 双节点 16× H800-80G (80GB VRAM)
- **精度**: BF16
- **并行**: TP=16, EP=16

### 内存配置
- **静态内存比例**: 70% (`mem_fraction_static=0.7`)
- **序列长度**: 34,816 tokens

---

## 关键文件索引

| 功能模块 | 文件路径 |
|---------|---------|
| 推理入口 | `longcat_omni_demo.py` |
| 视觉编码器 | `encoders/vision_adaptor.py` |
| 音频编码器 | `encoders/audio_adaptor.py` |
| 文本嵌入 | `encoders/embedding.py` |
| 数据处理 | `data/data_processor.py` |
| 多模态分词器 | `data/multimodal_tokenizer.py` |
| 后处理 | `post_process/unified_post_processor.py` |
| 常量定义 | `constants.py` |
| 全局配置 | `global_vars.py` |

---

## 关键设计亮点

### 1. 早期多模态融合
所有模态在 embedding 层就映射到统一空间，而非后期拼接。

### 2. 双重音频表示
- **连续音频**: DFSMN 处理原始波形，适合 ASR
- **离散 codec**: 4-codebook 高效编码，适合 TTS

### 3. 动态视觉压缩
`v_post_squeeze` 机制根据 `max_vision_length=28672` 动态调整。

### 4. 高效 MoE
560B 参数中仅激活 27B，激活率 ~4.8%。

---

## 许可

本分析文档基于 LongCat-Flash-Omni 开源代码编写。

## 相关链接

- [LongCat-Flash-Omni GitHub](https://github.com/meituan-longcat/LongCat-Flash-Omni)
- [SGLang](https://github.com/sgl-project/sglang)
