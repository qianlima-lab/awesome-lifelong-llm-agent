
# 0. Related Surveys:

## 0.1 Continual Learning in Computer Vison
- A comprehensive survey of continual learning: theory, method and application

## 0.2 Continual Learning in NLP
- Towards Lifelong Learning of Large Language Models: A Survey
- Continual learning for large language models: A survey
- Continual learning of large language models: A comprehensive survey

## 0.3 Agent Surveys:
- All Perspective:
  - Cognitive Architectures for Language Agents
  - Large Language Model based Multi-Agents: A Survey of Progress and Challenges
  - The rise and potential of large language model based agents: A survey
  - A survey on large language model based autonomous agents
  - An In-depth Survey of Large Language Model-based Artificial Intelligence Agents
  - Large language models empowered agent-based modeling and simulation: a survey and perspectives

- Memory Perspective:
  - A Survey on the Memory Mechanism of Large Language Model based Agents

- Planning Perspective:
  - Understanding the planning of LLM agents: A survey

- OS Perspective:
  - LLM as OS, Agents as Apps: Envisioning AIOS, Agents and the AIOS-Agent Ecosystem

# 1. Perception (How to ``Perceive`` Other Modalities from Environment into LLMs)

## 1.1 Single Modality (different environment and scenarios)
- To be determined

## 1.2 Multi Modality
- $\color{#FF8247}\text{MultiModal Continual Learning, CLIP, Audio, 3D}$
  - (a) Survey:
    - Recent Advances of Multimodal Continual Learning: A Comprehensive Survey
  - (b) Recent Papers:
    - LLMs Can Evolve Continually on Modality for X-Modal Reasoning
    - Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters
  - (c) Resources:
    - https://github.com/LucyDYu/Awesome-Multimodal-Continual-Learning


# 2. Memory (How to ``Memorize`` the Past Experimence and Knowledge)

## 2.1 Working Memory (input prompt)
- Prompt Compression
  - (a) Representative Papers:
    - LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models

- Long Context
  - (a) Representative Papers:
    - Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context
    - XLNet: Generalized Autoregressive Pretraining for Language Understanding
    - Lost in the Middle: How Language Models Use Long Contexts

- $\color{#FF8247}\text{Role Playing (Prompt Engingineering)}$
  - (a) Representative Papers:
    - Character-llm: A trainable agent for role-playing
    - IBSEN: Director-Actor Agent Collaboration for Controllable and Interactive Drama Script Generation
    - Theory of Mind for Multi-Agent Collaboration via Large Language Models 
  - (b) Survey:
    - Two tales of persona in llms: A survey of role-playing and personalization  


## 2.2 Episodic Memory (self experience)
- Traditional Continual Learning (with data replay technique)
  - (a) Recent Papers:
    - Contextual Experience Replay for Continual Learning of Language Agents (Submittied to ICLR 2025)

- $\color{#FF8247}\text{Continual RL (with data replay technique)}$
  - (a) Representative Papers:
    - Continual World: A Robotic Benchmark For Continual Reinforcement Learning
    - The Effectiveness of World Models for Continual Reinforcement Learning
  - (b) Survey:
    - Towards Continual Reinforcement Learning: A Review and Perspectives

- Lifelong Chat Agent
  - (a) Representative Papers:
    - Long Time No See! Open-Domain Conversation with Long-Term Persona Memory
    - Enhancing Large Language Model with Self-Controlled Memory Framework
    - Memory-assisted prompt editing to improve GPT-3 after deployment
    - MemoryBank: Enhancing Large Language Models with Long-Term Memory
  - (b) Resources:
    - https://modelscope.github.io/MemoryScope/zh/examples/api/simple_usages_zh.html  

- Web Agent
  - (a) Representative Papers:
    - Large Language Models Are Semi-Parametric Reinforcement Learning Agents


## 2.3 Semantic Memory (external resources)
- $\color{#FF8247}\text{Continual Knowledge Graphs}$
  - (a) Representative Papers:
    - Continual learning of knowledge graph embeddings
    - Towards continual knowledge graph embedding via incremental distillation
    - Continual multimodal knowledge graph construction
  - (b) Survey:
    - Continual Learning on Graphs: Challenges, Solutions, and Opportunities
    - Continual Learning on Graphs: A Survey 

- $\color{#FF8247}\text{Combined with Tool/RAG}$
  - (a) Representative Papers:
    - CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing
    - MemoRAG: Moving towards Next-Gen RAG Via Memory-Inspired Knowledge Discovery
    - Check Your Facts and Try Again: Improving Large Language Models with External Knowledge and Automated Feedback
    - Training Language Models with Memory Augmentation

## 2.4 LLM Memory (finetuning LLM parameters)
- $\color{#FF8247}\text{Self-Evolution}$
  - (a) Survey:
    - A Survey on Self-Evolution of Large Language Models

# 3. Action (How to ``Act`` in the Environment)

## 3.1 Grounding Action
- Tool
  - (a) Representative Papers:
    - ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings
    - LLMs in the Imaginarium: Tool Learning through Simulated Trial and Error
    - Large Language Models as Tool Makers
  - (b) Survey:
    - Tool Learning with Foundation Models
- Web
  - (a) Representative Papers:
    - WebArena: A Realistic Web Environment for Building Autonomous Agents
    - WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents
- Game
  - (a) Representative Papers:
    - Describe, Explain, Plan and Select: Interactive Planning with Large Language Models Enables Open-World Multi-Task Agents
    - Voyager: An Open-Ended Embodied Agent with Large Language Models

- Embodiedment
  - (a) Representative Papers:
    - VirtualHome: Simulating Household Activities Via Programs
    - ALFWorld: Aligning Text and Embodied Environments for Interactive Learning
    - Inner Monologue: Embodied Reasoning through Planning with Language Models

## 3.2 Retrival Action
- RAG
  - (a) Representative Papers:
    - Retrieval-augmented generation for knowledge-intensive NLP tasks


## 3.3 Reasoning Action
- LLM Reasoning
  - (a) Representative Papers:
    - ReAct: Synergizing Reasoning and Acting in Language Models
    - Tree of Thoughts: Deliberate Problem Solving with Large Language Models

## 3.4 Planning Action (or decision making)
- $\color{#FF8247}\text{LLM Planing}$
  - (a) Representative Papers:
    - Planning with Large Language Models via Corrective Re-prompting
    - Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents
    - Leveraging pre-trained large language models to construct and utilize world models for model-based task planning
  - (b) Survey:
    - Understanding the planning of LLM agents: A survey

- $\color{#FF8247}\text{Lifelong Robot Learning (Decision Making)}$
  - (a) Representative Papers:
    - LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning
    - Deploying Ten Thousand Robots: Scalable Imitation Learning for Lifelong Multi-Agent Path Finding
