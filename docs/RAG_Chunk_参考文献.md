# RAG 动态 Chunk 技术文章 · 参考文献

---

## 一、核心方法论文

### 1. Semantic Chunking（语义分块）

无单一原始论文，方法由 LangChain 和 LlamaIndex 工程社区实现并推广。
- LangChain 实现：`langchain_experimental.text_splitter.SemanticChunker`
- LlamaIndex 实现：`SemanticSplitterNodeParser`

---

### 2. Meta-Chunking / PPL Chunking

**标题：** Meta-Chunking: Learning Efficient Text Segmentation via Logical Perception
**作者：** Jihao Zhao, Zhiyuan Ji, Pengnian Qi, Simin Niu, Bo Tang, Feilong Tang, Zhiyu Li
**机构：** IAAR-Shanghai
**发表：** arXiv:2410.12788，2024年10月
**链接：** https://arxiv.org/abs/2410.12788
**核心贡献：** 提出以语言模型困惑度（PPL）信号检测文本逻辑边界，定义"元粒度"分块单元，并对比了 PPL Chunking 与 Margin Sampling Chunking 两种实现路径。

---

### 3. Late Chunking

**标题：** Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models
**作者：** Michael Günther, Isabelle Mohr, Bo Wang, Han Xiao et al.
**机构：** JinaAI
**发表：** arXiv:2409.04701，2024年9月；正式收录于 SIGIR 2025
**链接：** https://arxiv.org/abs/2409.04701
**核心贡献：** 提出"先全文嵌入、后按 span 切割并 mean pooling"的流程，使每个 chunk 嵌入携带全局文档上下文信息，从根本上解决孤立 chunk 的指代缺失问题。

---

### 4. Contextual Retrieval（Contextual Chunking）

**来源：** Anthropic 官方技术博客 + Cookbook
**发布时间：** 2024年9月
**链接：** https://www.anthropic.com/news/contextual-retrieval
**核心贡献：** 提出为每个 chunk 用 LLM 生成上下文前缀，并结合 Prompt Caching 将成本降至可接受水平；量化报告了 Contextual Embeddings + BM25 将检索失败率降低 49%。

---

### 5. Proposition-Based Chunking（Dense X Retrieval）

**标题：** Dense X Retrieval: What Retrieval Granularity Should We Use?
**作者：** Tong Chen, Hongwei Wang, Sihao Chen, Wenhao Yu, Kaixin Ma, Xinran Zhao, Hongming Zhang, Dong Yu
**机构：** 卡内基梅隆大学等
**发表：** arXiv:2312.06648，2023年12月；收录于 ACL 2024
**链接：** https://arxiv.org/abs/2312.06648
**核心贡献：** 提出将文本分解为原子级、自包含的事实"命题"作为检索单元，并设计了命题检索 + 段落聚合的两阶段流程。

---

### 6. RAPTOR

**标题：** RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval
**作者：** Parth Sarthi, Salman Abdullah, Aditi Goldie, Ashwin Sengupta, Kayla Khodamoradi, Sheryl Hsu, Eli Sherber, Emre Kıcıman, Hamid Palangi, Monica Lam, Percy Liang
**机构：** 斯坦福大学
**发表：** arXiv:2401.18059，2024年1月；收录于 ICLR 2024
**链接：** https://arxiv.org/abs/2401.18059
**核心贡献：** 提出自底向上递归构建摘要树的多层次检索架构，统一支持宏观摘要查询与细节查询，在 QuALITY 基准上与 GPT-4 结合提升 20%。

---

### 7. RAPTOR 增量更新变体（adRAP / postQFRAP）

**标题：** Retrieval-Augmented Generation with Dynamic Knowledge Graphs for Long-form Question Answering
**备注：** RAPTOR 动态更新变体，工程成熟度仍处于研究阶段
**发表：** arXiv:2410.01736，2024年10月
**链接：** https://arxiv.org/abs/2410.01736

---

### 8. HyDE（Hypothetical Document Embeddings）

**标题：** Precise Zero-Shot Dense Retrieval without Relevance Labels
**作者：** Luyu Gao, Xueguang Ma, Jimmy Lin, Jamie Callan
**机构：** 卡内基梅隆大学
**发表：** arXiv:2212.10496，2022年12月；收录于 ACL 2023
**链接：** https://arxiv.org/abs/2212.10496
**核心贡献：** 提出在检索前用 LLM 生成假设文档，将查询向量转换为文档风格向量，消除嵌入空间中查询与文档的风格鸿沟。

---

### 9. HyPE（Hypothetical Prompt Embeddings）

**标题：** HyPE: Hypothetical Prompt Embeddings for Retrieval-Augmented Generation
**作者：** Florian Vake, Jurij Vičič, Aleksander Tošić
**发表：** IEEE Access，2025年2月
**链接：** https://ieeexplore.ieee.org/document/10891988
**核心贡献：** 将假设内容生成从查询时移至索引时，对每个 chunk 预生成多个假设问题并存储其向量，实现查询零延迟的问题对问题匹配，精确率和召回率均有大幅提升。

---

### 10. Summary-Augmented Chunking（SAC）

**标题：** Summary-Augmented Chunking to Overcome Document-Level Retrieval Mismatch in RAG
**作者：** Nils Reuter, Johannes Reuter, Thorsten Schöler et al.
**发表：** arXiv:2510.06999，2025年10月；收录于 NLLP Workshop 2025（EMNLP）
**链接：** https://arxiv.org/abs/2510.06999
**核心贡献：** 定义了 DRM（Document-Level Retrieval Mismatch）失败模式，提出在 chunk 前拼接文档级摘要作为"文档指纹"，显著降低结构相似文档库中的跨文档混淆。

---

### 11. ColBERT / ColBERTv2

**标题（原版）：** ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT
**作者：** Omar Khattab, Matei Zaharia
**机构：** 斯坦福大学
**发表：** arXiv:2004.12832，2020年；收录于 SIGIR 2020
**链接：** https://arxiv.org/abs/2004.12832

**标题（v2）：** ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction
**作者：** Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, Matei Zaharia
**发表：** arXiv:2112.01488，2021年；收录于 NAACL 2022
**链接：** https://arxiv.org/abs/2112.01488
**核心贡献：** 提出 MaxSim late interaction 架构，为每个 token 保留独立向量，使检索匹配在 token 级别发生，绕开 chunk 粒度问题。

---

### 12. HOPE（Holistic Passage Evaluation）

**标题：** A New HOPE: Domain-agnostic Automatic Evaluation of Text Chunking
**作者：** Henrik Brådland, Morten Goodwin, Per-Arne Andersen, Alexander S. Nossum, Aditya Gupta
**发表：** arXiv:2505.02171，2025年5月；收录于 SIGIR 2025
**链接：** https://arxiv.org/abs/2505.02171
**核心贡献：** 提出 HOPE 指标，从 passage 内在属性、外在属性（passage 间关系）、与文档整体连贯性三个层次评估分块质量，在不依赖下游任务的前提下预测 RAG 性能。

---

## 二、基准与评测研究

### 13. Vectara Chunking & Embedding Benchmark（NAACL 2025）

**标题：** CRAG: A Benchmark for Evaluating Chunking and Retrieval in Augmented Generation
**机构：** Vectara
**发表：** arXiv:2410.13070，2024年10月；收录于 NAACL 2025 Findings
**链接：** https://arxiv.org/abs/2410.13070
**核心结论：** 跨 25 种分块配置与 48 个嵌入模型的系统性测评，分块配置对检索质量的影响等于甚至超过嵌入模型选择；MTEB 分数无法预测领域特定性能。

---

### 14. FloTorch 2026 Chunking Benchmark

**来源：** FloTorch 团队，2026年初
**核心结论：** 在 50 篇学术论文（905,746 token）上测试 7 种分块策略，递归字符分割（512 token）以 69% 端到端准确率位列第一，语义分块以 54% 垫底（平均 chunk size 仅 43 token，LLM 上下文不足）。

---

### 15. Chroma 上下文衰减研究（Context Rot）

**来源：** Chroma 研究团队，2025年7月
**核心结论：** 跨 GPT-4.1、Claude 4、Gemini 2.5 等 18 个模型，检索性能在约 2,500 token 处出现断崖式下降（"上下文悬崖"），为 chunk size 设定了工程上限。

---

### 16. MDPI Bioengineering 临床决策支持研究

**标题：** Adaptive Chunking for Clinical Decision Support Systems Using RAG
**发表：** MDPI Bioengineering，2025年11月
**核心结论：** 对齐逻辑主题边界的自适应分块在临床文档上达到 87% 准确率，固定大小基线仅 13%（p=0.001），在特定领域分块策略的影响远超通用基准的结论。

---

## 三、工具库与工程参考

### 17. RAGAS（RAG 评估框架）

**项目地址：** https://github.com/explodinggradients/ragas
**用途：** 提供 Context Precision、Context Recall、Faithfulness、Answer Correctness 等 RAG 专用评估指标，支持从文档自动生成合成测试集，兼容 LangChain、LlamaIndex、Haystack。

---

### 18. RAGatouille（ColBERT 工程化库）

**项目地址：** https://github.com/bclavie/RAGatouille
**用途：** 将 ColBERTv2 工程化适配 RAG 场景，内置文档分块，提供简洁的索引与检索 API。

---

### 19. Microsoft Azure Architecture Center RAG 指南

**来源：** Microsoft Azure Architecture Center，2025年
**链接：** https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-chunking-phase
**核心建议：** 512 token + 128 token 重叠作为生产起点，使用 BERT token 计数；元数据注入（文档标题、章节、日期）可将 QA 准确率从 50–60% 提升至 72–75%。

---
