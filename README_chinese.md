# Lifelong Learning of Large Language Model based Agents: A Roadmap

[![arXiv](https://img.shields.io/badge/arXiv-lifelong_LLM_Agents-b31b1b.svg)](https://arxiv.org/pdf/2501.07278)

欢迎来到与我们综述论文 **Lifelong Learning of Large Language Model based Agents: A Roadmap** 对应的仓库。本仓库收集了与 LLM agent 的终生学习（也称为持续学习、增量学习）相关的优秀论文。我们在论文中指出，感知（Perception）、记忆（Memory）和行动（Action）三个关键模块是实现 LLM agent 持续学习能力的核心所在。详情可参考[这篇综述论文](https://arxiv.org/pdf/2501.07278)。另外，关于LLM的终生学习（持续学习、增量学习）的其他论文，综述以及资源，可以参考这个[仓库](https://github.com/zzz47zzz/awesome-lifelong-learning-methods-for-llm)。英文版本的README提供在这个[文件](./README.md).

![illustrution](./assets/illustrution_chinese.png)

![introduction](./assets/introduction_chinese.png)

## 📢 最新动态

- **2025-1-14**: 我们发布了综述论文「[Lifelong Learning of Large Language Model based Agents: A Roadmap](https://arxiv.org/pdf/2501.07278)」。欢迎引用或提交 pull request。

## 📒 目录


## 感知模块 (Perception Module)

### 单模态感知 (Single-Modal Perception)

| 标题 | 会议/期刊 | 日期 |
|:---|:---|:---|
|[AgentOccam: A Simple Yet Strong Baseline for LLM-Based Web Agent](https://arxiv.org/pdf/2410.13825?)|arXiv|2024-10|
|[GPT-4V(ision) is a Generalist Web Agent, if Grounded](https://arxiv.org/pdf/2401.01614.pdf)|ICLR|2024-01|
|[Webarena: A realistic web environment for building autonomous agents](https://arxiv.org/pdf/2307.13854)|ICLR|2023-07|
|[Synapse: Trajectory-asexemplar prompting with memory for computer control](https://openreview.net/pdf?id=Pc8AU1aF5e)|ICLR|2023-06|
|[Multimodal web navigation with instruction-finetuned foundation models](https://arxiv.org/pdf/2305.11854)|ICLR|2023-05|

### 多模态感知 (Multi-Modal Perception)

| 标题 | 会议/期刊 | 日期 |
|:---|:---|:---|
|[Llms can evolve continually on modality for x-modal reasoning](https://arxiv.org/pdf/2410.20178)|NeurIPS|2024-10|
|[Modaverse: Efficiently transforming modalities with llms](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_ModaVerse_Efficiently_Transforming_Modalities_with_LLMs_CVPR_2024_paper.pdf)|CVPR|2024-01|
|[Omnivore: A single model for many visual modalities](https://arxiv.org/PDF/2201.08377)|CVPR|2022-01|
|[Perceiver: General perception with iterative attention](http://proceedings.mlr.press/v139/jaegle21a/jaegle21a.pdf)|ICML|2021-07|
|[Vatt: Transformers for multimodal self-supervised learning from raw video, audio and text](https://proceedings.neurips.cc/paper_files/paper/2021/file/cb3213ada48302953cb0f166464ab356-Paper.pdf)|NeurIPS|2021-04|

## 记忆模块 (Memory Module)

### 工作记忆 (Working Memory)

| 标题 | 会议/期刊 | 日期 |
|:---|:---|:---|
|[Character-llm: A trainable agent for role-playing](https://arxiv.org/pdf/2310.10158)|EMNLP|2023-10|
|[Connecting Large Language Models with Evolutionary Algorithms Yields Powerful Prompt Optimizers](https://arxiv.org/pdf/2309.08532)|ICLR|2023-09|
|[Adapting Language Models to Compress Contexts](https://arxiv.org/pdf/2305.14788)|ACL|2023-05|
|[Critic: Large language models can self-correct with tool-interactive critiquing}](https://arxiv.org/pdf/2305.11738)|NeurIPS Workshop|2023-05|
|[Cogltx: Applying bert to long texts](https://proceedings.neurips.cc/paper_files/paper/2020/file/96671501524948bc3937b4b30d0e57b9-Paper.pdf)|NeurIPS|2020-12|

### 情景记忆 (Episodic Memory)

| 标题 | 会议/期刊 | 日期 |
|:---|:---|:---|
|[EmbodiedMemory-Bench: Benchmarking Embodied Memory for Long-Horizon Embodied Tasks](https://arxiv.org/abs/2609.28236)|arXiv|2026-09|
|[DelTA: An Online Document-Level Translation Agent Based on Multi-Level Memory](https://arxiv.org/pdf/2410.08143)|arXiv|2024-10|
|[MemoChat: Tuning LLMs to Use Memos for Consistent Long-Range Open-Domain Conversation](https://arxiv.org/pdf/2308.08239)|arXiv|2023-08|
|[RET-LLM: Towards a General Read-Write Memory for Large Language Models](https://arxiv.org/abs/2305.14322)|ICLR|2023-05|
|[Bring Evanescent Representations to Life in Lifelong Class Incremental Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Toldo_Bring_Evanescent_Representations_to_Life_in_Lifelong_Class_Incremental_Learning_CVPR_2022_paper.pdf)|CVPR|2022-06|
|[iCaRL: Incremental Classifier and Representation Learning](https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf)|CVPR|2017-04|

### 语义记忆 (Semantic Memory)

| 标题 | 会议/期刊 | 日期 |
|:---|:---|:---|
|[Fast and Continual Knowledge Graph Embedding via Incremental LoRA](https://arxiv.org/pdf/2407.05705?)|IJCAL|2024-07|
|[PromptDSI: Prompt-based Rehearsal-free Instance-wise Incremental Learning for Document Retrieval](https://arxiv.org/pdf/2406.12593)|arXiv|2024-06|
|[CorpusBrain++: A Continual Generative Pre-Training Framework for Knowledge-Intensive Language Tasks](https://arxiv.org/pdf/2402.16767)|arXiv|2024-02|
|[Continual Multimodal Knowledge Graph Construction](https://arxiv.org/pdf/2305.08698)|IJCAI|2023-05|
|[Lifelong embedding learning and transfer for growing knowledge graphs](https://ojs.aaai.org/index.php/AAAI/article/view/25539/25311)|AAAI|2022-11|

### 参数记忆 (Parametric Memory)

| 标题 | 会议/期刊 | 日期 |
|:---|:---|:---|
|[ELDER: Enhancing Lifelong Model Editing with Mixture-of-LoRA](https://arxiv.org/pdf/2408.11869)|arXiv|2024-08|
|[WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models](https://arxiv.org/pdf/2405.14768)|NeurIPS|2024-05|
|[WilKE: Wise-Layer Knowledge Editor for Lifelong Knowledge Editing](https://arxiv.org/pdf/2402.10987)|ACL|2024-02|
|[Aging with GRACE: Lifelong Model Editing with Key-Value Adaptors](https://openreview.net/pdf?id=ngCT1EelZk)|ICLR|2022-11|
|[Plug-and-Play Adaptation for Continuously-updated QA](https://arxiv.org/pdf/2204.12785)|ACL|2022-04|

## 行动模块 (Action Module)

### 基础行动 (Grounding Actions)

#### 工具环境 (Tool Environment)

| 标题 | 会议/期刊 | 日期 |
|:---|:---|:---|
|[LLMs in the Imaginarium: Tool Learning through Simulated Trial and Error](https://aclanthology.org/2024.acl-long.570.pdf)|ACL|2024-03|
|[EASYTOOL: Enhancing LLM-based Agents with Concise Tool Instruction](https://arxiv.org/pdf/2401.06201)|ICLR Workshop|2024-01|
|[Confucius: Iterative Tool Learning from Introspection Feedback by Easy-to-Difficult Curriculum](https://ojs.aaai.org/index.php/AAAI/article/view/29759)|AAAI|2023-08|
|[GEAR: Augmenting Language Models with Generalizable and Efficient Tool Resolution](https://arxiv.org/pdf/2307.08775)|EACL|2023-07|
|[ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs](https://arxiv.org/pdf/2307.16789)|ICLR|2023-07|
|[Large Language Models as Tool Makers](https://arxiv.org/pdf/2305.17126)|ICLR|2023-05|
|[Toolformer: Language Models Can Teach Themselves to Use Tools](https://openreview.net/pdf?id=Yacmpz84TH)|NeurIPS|2023-05|
|[ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings](https://openreview.net/pdf?id=BHXsb69bSx)|NeurIPS|2023-05|
|[On the Tool Manipulation Capability of Open-source Large Language Models](https://arxiv.org/pdf/2305.16504)|arXiv|2023-05|
|[ART: Automatic multi-step reasoning and tool-use for large language models](https://arxiv.org/pdf/2303.09014)|arXiv|2023-03|

#### Web 环境 (Web Environment)

| 标题 | 会议/期刊 | 日期 |
|:---|:---|:---|
|[AgentOccam: A Simple Yet Strong Baseline for LLM-Based Web Agent](https://arxiv.org/pdf/2410.13825?)|arXiv|2024-10|
|[WebPilot: A Versatile and Autonomous Multi-Agent System for Web Task Execution with Strategic Exploration](https://arxiv.org/pdf/2408.15978)|arXiv|2024-08|
|[SteP: Stacked LLM Policies for Web Actions](https://openreview.net/pdf?id=5fg0VtRxgi)|COLM|2024-07|
|[LASER: LLM Agent with State-Space Exploration for Web Navigation](https://arxiv.org/pdf/2309.08172)|NeurIPS Workshop|2023-09|
|[Large Language Models Are Semi-Parametric Reinforcement Learning Agent](https://proceedings.neurips.cc/paper_files/paper/2023/file/f6b22ac37beb5da61efd4882082c9ecd-Paper-Conference.pdf)|NeurIPS|2023-06|

#### 游戏环境 (Game Environment)

| 标题 | 会议/期刊 | 日期 |
|:---|:---|:---|
|[VillagerAgent: A Graph-Based Multi-Agent Framework for Coordinating Complex Task Dependencies in Minecraft](https://arxiv.org/pdf/2406.05720)|ACL|2024-06|
|[See and Think: Embodied Agent in Virtual Environment](https://arxiv.org/pdf/2311.15209)|ECCV|2023-11|
|[JARVIS-1: Open-World Multi-task Agents with Memory-Augmented Multimodal Language Models](https://arxiv.org/pdf/2311.05997)|TPAMI|2023-11|
|[Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/pdf/2305.16291)|arXiv|2023-05|
|[Describe, Explain, Plan and Select: Interactive Planning with Large Language Models Enables Open-World Multi-Task Agents](https://arxiv.org/pdf/2302.01560)|NeurIPS|2023-02|

### 检索行动 (Retrieval Actions)

#### 从语义记忆中检索 (Retrieval from Semantic memory)

| 标题 | 会议/期刊 | 日期 |
|:---|:---|:---|
|[See and Think: Embodied Agent in Virtual Environment](https://arxiv.org/pdf/2311.15209)|ECCV|2023-11|
|[Ghost in the Minecraft: Generally Capable Agents for Open-World Environments via Large Language Models with Text-based Knowledge and Memory](https://arxiv.org/pdf/2305.17144)|arXiv|2023-05|
|[Planning with Large Language Models via Corrective Re-prompting](https://openreview.net/pdf?id=cMDMRBe1TKs)|NeurIPS Workshop|2022-10|
|[Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents](https://proceedings.mlr.press/v162/huang22a/huang22a.pdf)|ICML|2022-01|

#### 从情景记忆中检索 (Retrieval from Episodic memory)

| 标题 | 会议/期刊 | 日期 |
|:---|:---|:---|
|[VLM Agents Generate Their Own Memories: Distilling Experience into Embodied Programs](https://arxiv.org/pdf/2406.14596)|arXiv|2024-06|
|[Large Language Models as Tool Makers](https://arxiv.org/pdf/2305.17126)|ICLR|2023-05|
|[On the Tool Manipulation Capability of Open-source Large Language Models](https://arxiv.org/pdf/2305.16504)|arXiv|2023-05|
|[Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/pdf/2305.16291)|arXiv|2023-05|
|[ART: Automatic multi-step reasoning and tool-use for large language models](https://arxiv.org/pdf/2303.09014)|arXiv|2023-03|

### 推理行动 (Reasoning Actions)

#### 会话内记忆推理 (Intra-Episodic Memory)

| 标题 | 会议/期刊 | 日期 |
|:---|:---|:---|
|[Reasoning with Language Model is Planning with World Model](https://arxiv.org/pdf/2305.14992)|EMNLP|2023-05|
|[Large Language Models as Commonsense Knowledge for Large-Scale Task Planning](https://proceedings.neurips.cc/paper_files/paper/2023/file/65a39213d7d0e1eb5d192aa77e77eeb7-Paper-Conference.pdf)|NeurIPS|2023-05|
|[Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2023/file/271db9922b8d1f4dd7aaef84ed5ac703-Paper-Conference.pdf)|NeurIPS|2023-05|
|[SwiftSage: A Generative Agent with Fast and Slow Thinking for Complex Interactive Tasks](https://proceedings.neurips.cc/paper_files/paper/2023/file/4b0eea69deea512c9e2c469187643dc2-Paper-Conference.pdf)|NeurIPS2023|2023-05|
|[Reflexion: Language Agents with Verbal Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2023/file/1b44b878bb782e6954cd888628510e90-Paper-Conference.pdf)|NeurIPS|2023-03|
|[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/pdf/2210.03629)|ICLR|2022-10|

#### 会话间记忆推理 (Inter-Episodic Memory)

| 标题 | 会议/期刊 | 日期 |
|:---|:---|:---|
|[VLM Agents Generate Their Own Memories: Distilling Experience into Embodied Programs](https://arxiv.org/pdf/2406.14596)|arXiv|2024-06|
|[LLMs in the Imaginarium: Tool Learning through Simulated Trial and Error](https://aclanthology.org/2024.acl-long.570.pdf)|ACL|2024-03|
|[See and Think: Embodied Agent in Virtual Environment](https://arxiv.org/pdf/2311.15209)|ECCV|2023-11|
|[Large Language Models Are Semi-Parametric Reinforcement Learning Agent](https://proceedings.neurips.cc/paper_files/paper/2023/file/f6b22ac37beb5da61efd4882082c9ecd-Paper-Conference.pdf)|NeurIPS|2023-06|
|[Ghost in the Minecraft: Generally Capable Agents for Open-World Environments via Large Language Models with Text-based Knowledge and Memory](https://arxiv.org/pdf/2305.17144)|arXiv|2023-05|
|[Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/pdf/2305.16291)|arXiv|2023-05|

## 📚 引用我们的工作

```bibtex
@article{zheng2025lifelong,
      title={Lifelong Learning of Large Language Model based Agents: A Roadmap}, 
      author={Zheng, Junhao and Shi, Chengming and Cai, Xidi and Li, Qiuke and Zhang, Duzhen and Li, Chenxing and Yu, Dong and Ma, Qianli},
      journal={arXiv preprint arXiv:2501.07278},
      year={2025},
}
```

## 😍 星星历史

<a href="https://star-history.com/#qianlima-lab/awesome-lifelong-llm-agent&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=qianlima-lab/awesome-lifelong-llm-agent&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=qianlima-lab/awesome-lifelong-llm-agent&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=qianlima-lab/awesome-lifelong-llm-agent&type=Date" />
 </picture>
</a>