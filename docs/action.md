# Action (How to `Act` in the Environment)

## Grounding Action

**Definition**: Grounding procedures execute external actions and process environmental feedback into working memory as text. This effectively simplifies the agent’s interaction with the outside world as a “text game” with textual observations and actions.

环境从简单到复杂可以分为tool, web, game, embodiment. Tool的输入空间比较简单, 通常只有工具的描述以及instruction. Web的输入空间是网页的无障碍树 (或HTML页面), 可执行的action, instruction. Game和embodiment的输入空间更加复杂. Game的输入空间通常是更加现实的, 复杂的场景的文字描述. Embodiment和game比较像, 因为游戏和现实彼此之间的复杂程度比较接近. 但是embodiment的输入空间一般涉及更多的模态 (e.g., 视觉).

==Adaption的角度==

**从LLM input的角度论述**: 在预训练时, corpus中大多是整段的文本. 而agent收到的环境描述, 以及环境反馈大多是简短的句子和短语, 或者存在大量预训练corpus中出现频率较低的文本类型 (e.g., JSON, HTML label). Agent需要学会如何在新的环境中执行任务, 对于LLM来说, 其需要学会更好地理解来自环境的输入. 通过学习环境输入的角度和连续学习建立关联.

**从LLM output的角度论述**: 其还需要通过输出特定模式字符串等方式执行action. 通过学习如何向环境输出的角度和连续学习建立关联. LLM原本只能输出文字内容, 现在其能够执行更加高级的动作 (e.g., 调用工具, 浏览网页), 这实际上体现了其能力的增加. 通过能力增加的角度和连续学习建立关联.

> [!NOTE]
> 不包含注重策略的文章 (e.g., 如何选择更加合适的工具). 这些文章更加适合planning和reasoning部分.

对于每一种环境, 分别从文章中总结环境中独自的挑战, 以及其是从哪个角度来进行LLM能力的增加. (记录挑战以及强相关文章)

### Tool

- Input space
  - GEAR: Augmenting Language Models with Generalizable and Efficient Tool Resolution (EACL 2024): 通过小参数量的LM计算tool和query的匹配程度, 为LLM选择合适的工具.
  - ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs (ICLR 2024 spotlight): 在进行DFSDT搜索的过程中, 之前的工具调用轨迹可以帮助模型了解有问题的工具, 学习工具的调用方式.
  - EASYTOOL: Enhancing LLM-based Agents with Concise Tool Instruction (ICLR 2024 Workshop): 文章提出通过ChatGPT将多样的, 长的工具文档转化为统一的, 准确的tool instruction, 方便LLM判断要调用哪些工具, 并为该工具传入哪些参数.
  - ART: Automatic multi-step reasoning and tool-use for large language models (arxiv 2023): 根据用户输入从数据库中选择人工编写或者改写的工具调用轨迹, 教会LLM进行工具调用.
  - On the Tool Manipulation Capability of Open-source Large Language Models (FMDM@NeurIPS2023): 通过微调以及检索工具调用例子, 教会LLM进行工具调用.
  - LLMs in the Imaginarium: Tool Learning through Simulated Trial and Error (ACL 2024): 根据GPT生成的工具调用轨迹进行LLM的微调或者in-context-learning.
  - Confucius: Iterative Tool Learning from Introspection Feedback by Easy-to-Difficult Curriculum (AAAI 2024): 微调LLM使得其能够更好地理解工具能力, 进行工具的调用.
  - Large Language Models as Tool Makers (ICLR 2024): 通过GPT4制作的文档, 帮助参数量更小的LLM使用工具.
- Output space
  - ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings (NeurIPS 2023 ora)): 通过toollken来学习新的工具.
  - Toolformer: Language Models Can Teach Themselves to Use Tools (NeurIPS 2023 oral): 通过微调LLM的方式使得LLM具有了调用外界工具的能力. 调用的方式为输出格式的字符串.
  - Large Language Models as Tool Makers (ICLR 2024): GPT4制作工具本身, 供小LLM调用.
- Other:
  - ToolNet: Connecting Large Language Models with Massive Tools via Tool Graph (arxiv 2024)
  - Small LLMs Are Weak Tool Learners: A Multi-LLM Agent (submitted to ACL 2024)
- Benchmark and dataset
  - ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs (ICLR 2024 spotlight)
  - StableToolBench: Towards Stable Large-Scale Benchmarking on Tool Learning of Large Language Models (ACL 2024 findings)
  - Gorilla: Large Language Model Connected with Massive APIs (arxiv 2023)
  - ToolAlpaca: Generalized Tool Learning for Language Models with 3000 Simulated Cases (arxiv 2023)
  - On the Tool Manipulation Capability of Open-source Large Language Models (FMDM@NeurIPS2023)
  - API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs (EMNLP 2023)
- Strongly related
  - ART: Automatic multi-step reasoning and tool-use for large language models (arxiv 2023): LLM生成的工具调用轨迹在人工改写后会被放入数据库.
  - LLMs in the Imaginarium: Tool Learning through Simulated Trial and Error (ACL 2024): 文章认为在连续学习的过程中, 通过简单的replay strategy, 便可以有效克服灾难性遗忘的问题.
  - Confucius: Iterative Tool Learning from Introspection Feedback by Easy-to-Difficult Curriculum (AAAI 2024): 在LLM微调的过程中, 不断根据LLM的训练结果, 选择那些LLM难以学会的工具, 迭代地更新用于微调LLM的数据集.
  - Large Language Models as Tool Makers (ICLR 2024): 当LLM (e.g., GPT-3.5 Turbo) 遇到过多同类问题时, 其会通过其他更强大的LLM (e.g., GPT-4) 制作能够解决这些问题的工具, 并将之这些工具放入工具库中, 用于解决之后的问题.
  - Learning Evolving Tools for Large Language Models (arxiv 2024): 动态环境中LLM可以调用的工具可能会随着时间推移而过时. LLM的prompt中可能无法提供工具的准确信息, 如名称, 参数等. 文章让LLM通过环境反馈更新prompt中工具的用法.


### Web

- Input space
  - SteP: Stacked LLM Policies for Web Actions (COLM 2024): 人工进行策略的制定, 帮助LLM更好地完成任务. 在LLM执行在执行一个策略时, prompt中只需要包含策略相关的信息, 从而更加高效. 较短的prompt有利于LLM提取出关键信息.
  - AgentOccam: A Simple Yet Strong Baseline for LLM-Based Web Agents (submitted to ICLR 2025): 进行了HTML元素的简化, 通过树形结构选择性地将网页内容添加到prompt中, 删除prompt中不重要的observations.
  - WebPilot: A Versatile and Autonomous Multi-Agent System for Web Task Execution with Strategic Exploration (arxiv 2024): 通过收集reflection, 借助MCTS帮助LLM选择正确的action.
  - Large Language Models Are Semi-Parametric Reinforcement Learning Agents (NeurIPS 2023): 将来自多个任务的交互经验综合起来进行更好的决策 (即选择正确的action).
  - LASER: LLM Agent with State-Space Exploration for Web Navigation (FMDM@NeurIPS2023): 将任务建模为state-space exploration, LLM通过在每一个state内进行搜索, 在不同state间转移完成任务. 每一个state有更加专一的prompt, 帮助LLM更好地了解当前状态可以执行的action.
  - VLM Agents Generate Their Own Memories: Distilling Experience into Embodied Programs (NeurIPS 2024 Spotlight): 利用VLM本身, 在人类帮助下, 从次优演示中生成能够用于上下文学习的演示. 演示被存储在数据库中, 在VLM进行推理时, 其通过RAG从数据库中检索出最相关的轨迹加在prompt中.
  - Synapse: Trajectory-as-Exemplar Prompting with Memory for Computer Control (ICLR 2024): 将状态 (网页的HTML源码进行) 化简后输入到LLM中, 使得prompt中可以包括之前任务的完整轨迹做为示例.
- Output space
  - SteP: Stacked LLM Policies for Web Actions (COLM 2024): 通过代码阅读确定了每一个策略额外指定了agent可以执行的action.
  - AgentOccam: A Simple Yet Strong Baseline for LLM-Based Web Agents (submitted to ICLR 2025): 修改了WebArena中预先定义的动作, 重新定义哪些行为是agent可以做的, 哪些是不能做的. 
  - LASER: LLM Agent with State-Space Exploration for Web Navigation (FMDM@NeurIPS2023): 将任务建模为state-space exploration, 为每一个state定制了action space.
- Other
  - ADaPT: As-Needed Decomposition and Planning with Language Models (NAACL 2024 (findings))
  - Tree Search for Language Model Agents (submitted to ICLR 2025)
- Benchmark and dataset
  - WebArena: A Realistic Web Environment for Building Autonomous Agents (ICLR 2024)
  - WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents (NeurIPS 2022)
- Strongly related
  - Large Language Models Are Semi-Parametric Reinforcement Learning Agents (NeurIPS 2023): 使得LLM能够根据过去的交互经验进行更好的决策. 将来自多个任务的交互经验综合起来进行更好的决策.
  - VLM Agents Generate Their Own Memories: Distilling Experience into Embodied Programs (NeurIPS 2024 Spotlight): 利用VLM本身, 在人类帮助下, 从次优演示中生成能够用于上下文学习的演示. 演示被存储在数据库中, 在VLM进行推理时, 其通过RAG从数据库中检索出最相关的轨迹加在prompt中. 随着存储的轨迹逐渐增减VLM的性能也得到了提升.
  - Synapse: Trajectory-as-Exemplar Prompting with Memory for Computer Control (ICLR 2024): 将状态 (网页的HTML源码进行) 化简后输入到LLM中, 使得prompt中可以包括之前任务的完整轨迹做为示例.

### Game

- Input space
  - Describe, Explain, Plan and Select: Interactive Planning with Large Language Models Enables Open-World Multi-Task Agents (NeurIPS 2023 poster): 通过多个LLM (descriptor和explainer) 来更好地处理环境的反馈.
  - JARVIS-1: Open-World Multi-task Agents with Memory-Augmented Multimodal Language Models (arxiv 2023): 通过自我反思, 自我解释等方式更好地理解环境. 将之前的计划加在prompt中.
  - VillagerAgent: A Graph-Based Multi-Agent Framework for Coordinating Complex Task Dependencies in Minecraft (ACL 2024 findings): 通过独立的state manager模块, 从全局环境中过滤出任务相关的环境信息.
  - See and Think: Embodied Agent in Virtual Environment (ECCV 2024): 通过视觉编码模块为LLM提供更加丰富的环境信息.
  - Cradle: Empowering Foundation Agents Towards General Computer Control (NeurIPS 2024): 直接使用多模态大模型 (LMM) 而不是LLM.
- Output space
  - Voyager: An Open-Ended Embodied Agent with Large Language Models (arxiv 2023): Voyage通过不断学习的新技能与环境交互.
  - See and Think: Embodied Agent in Virtual Environment (ECCV 2024): 通过人工预先编写好的代码与环境交互.
  - Ghost in the Minecraft: Generally Capable Agents for Open-World Environments via Large Language Models with Text-based Knowledge and Memory (arxiv 2023): 利用一个单独的LLM将动作映射为键盘/鼠标操作.
  - Cradle: Empowering Foundation Agents Towards General Computer Control (NeurIPS 2024): 利用LMM生成代码控制键盘/鼠标.
- Other
- Benchmark and dataset
- Strongly related
  - Voyager: An Open-Ended Embodied Agent with Large Language Models (arxiv 2023): Voyager是第一个由LLM驱动的, 具有终身学习能力的Minecraft agent.
  - JARVIS-1: Open-World Multi-task Agents with Memory-Augmented Multimodal Language Models (arxiv 2023): 将之前的计划存储在multimodal memory中, 在执行新任务时进行召回.
  - See and Think: Embodied Agent in Virtual Environment (ECCV 2024): 通过课程学习让agent获得经验, 并通过chain of summarization将这些经验浓缩后添加在agent的上下文中, 用于之后的计划生成.
  - Cradle: Empowering Foundation Agents Towards General Computer Control (NeurIPS 2024): 将过去的经验和生成的用于和环境的代码存储在memory中, 用于和环境交互.

## Retrieval Action

**Definition**: A retrieval procedure reads information from long-term memories into working memory. 

在连续学习的过程中, LLM需要不断进行信息的更新 (信息包括从以往的任务中总结到的知识, 或者LLM在预训练之后才出现的新信息). 使用RAG可以低成本地为LLM带来新的信息.

额外的资料可以为LLM带来更加新的信息. 不断更新的信息可以使得LLM的能力不断提升 (或者不退化). 通过信息更新 (即信息的存储) 和连续学习建立关联. 

根据需要被存入数据库信息的来源对retrieval action进行细分. 信息的来源有两类, 分别是episodic memory (e.g., lifelong chat agent不断存储用户的信息) 和semantic memory (不断更新文本资料库?). 阅读这两类文章, 每一类文章中分别针对存在的挑战进行细分.

> [!NOTE] 
> 这里似乎没有办法和连续学习建立更强的关联. RAG数据库的增量被归类在了Semantic Memory一节. 

### Semantic memory

- See and Think: Embodied Agent in Virtual Environment (ECCV 2024)
  - 检索方式: 向量相似度.
  - 检索目的: 使得agent可以利用代码在虚拟环境中行动.
  - Query: LLM生成的, 文字形式的动作对应的embedding.
  - Key: 由ChatGPT生成的代码片段的描述对应的embedding.
  - Value: 人工预先编写好的代码, 通过执行这些代码, agent可以在虚拟环境中行动.
- Ghost in the Minecraft: Generally Capable Agents for Open-World Environments via Large Language Models with Text-based Knowledge and Memory (arxiv 2023)
  - 检索方式: 向量相似度.
  - 检索目的: 在进行任务分解时, 框架从数据库中检索出text-based knowledge, 用于子目标的分解.
  - Query: 当前的任务目标对应的embedding.
  - Key: Text-based knowledge对应的embedding.
  - Value: 来自互联网 (Minecraft Wiki) 的text-based knowledge的embedding.
- Planning with Large Language Models via Corrective Re-prompting (FMDM@NeurIPS2022)
  - 检索方式: 向量相似度.
  - 检索目的: 让LLM按照制定格式生成计划. 检索最相似的任务来提高生成任务的质量.
  - Query: 当前的任务对应的embedding.
  - Key: 数据库中任务的embedding.
  - Value: 数据库中的任务描述及其解决方案. 数据库中的任务并不会实时更新, 应该是人工预先编写的 (或来自训练集).
- Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents (ICML 2022): 
  - 检索方式: 向量相似度.
  - 检索目的: LLM直接生成的agent的行为可能不能被框架理解, 转化为agent的动作. 通过Sentence-BERT, 将LLM生成的句子转化为agent所有可能执行的动作中, 和LLM生成的动作语义最相似的那个.
  - Query: 当前LLM生成的步骤 (即agent的action) 对应的embedding.
  - Key: 当前可以执行的所有行为的embedding.
  - Value: 当前可以执行的所有行为.
- Corrective Retrieval Augmented Generation (arxiv 2024)
  - 检索方式: 文章主要侧重对检索的结果进行更正, 没有限定检索方式.
  - 检索目的: 通过检索得到的文档更好地回答问题. 文章主要侧重对检索的结果进行更正, 滤除文档中的不相干信息. 文章可以用于说明RAG的有效性, RAG中相关文档的重要性.
  - Query: 用户的问题.
  - Key: 文档.
  - Value: 文档.

- Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection (ICLR 2024 (oral)

  - 检索方式: Self-RAG框架强调对检索到的内容和生成结果进行反思, 并没有限定检索方式.

  - 检索目的: 通过检索出相关文档, 给出正确的生成结果. 在生成答案的过程中, 框架会决定什么时候要进行检索, 并对通过检索生成的答案进行评估. 可以用于说明相关文档的重要性.

  - Query: 任务输入.

  - Key: 短文 (passage).

  - Value: 短文 (passage).

### Episodic memory

- ART: Automatic multi-step reasoning and tool-use for large language models (arxiv 2023)
  - 检索方式: 文章将task library分成了5个cluster. 文章有两种策略为新任务构造prompt.
    - 对于有标注数据, 遍历5个cluster, 对于每一个cluster, 使用其中的工具调用轨迹构造prompt, 通过有标注数据判断哪一个cluster对应的prompt性能最好, 进而确定在无标注数据上使用的prompt. 
    - 对于无标注数据, 遍历task library中的所有旧任务, 通过LLM判断新任务和旧任务的相似程度, 使用最相似的旧任务对应的工具调用轨迹构造prompt. 具体来说, 文章通过LLM在推理时 “Similar” 和 “Not similar” 两个词的log probability ratio来找出最相关的工具.
  - 检索目的: 将从task library中检索出来的工具调用轨迹加在prompt中, 告诉LLM如何调用工具. Task library一开始存储的是人工编写的工具调用轨迹. 只后, 由LLM生成的, 经过人工改写的工具调用轨迹也会被添加到task library中. 在改写工具调用轨迹的过程中, 人类可以纠正LLM的错误, 或者添加新工具的调用. 改写后的工具调用轨迹会被放入到task library中.
  - Query: 参阅 "检索方式".
  - Key: 参阅 "检索方式".
  - Value: Task library中的工具调用轨迹.
- On the Tool Manipulation Capability of Open-source Large Language Models (FMDM@NeurIPS2023)
  - 检索方式: BM25
  - 检索目的: 将检索出来的demonstration加在prompt中, 告诉LLM如何调用工具.
  - Query: 目标描述 (goal description, 应该就是instruction).
  - Key: 有关API调用的demonstration.
  - Value: 人工编写的, 展示如何进行API调用的demonstration.
- Large Language Models as Tool Makers (ICLR 2024): 通过GPT4根据收集的问题制作能够解决问题的API, 放入工具仓库中.
  - 检索方式: 将一个使用特定prompt模板的LLM做为dispatcher, 有dispatcher根据工具的文档和question挑选合适的工具文档.
  - 检索目的: 工具仓库中存储着由GPT4制作的工具及工具文档, 将检索出来的工具文档加在prompt中, 告诉LLM如何进行工具的调用.
  - Query: Question.
  - Key: 工具文档.
  - Value: 工具文档.
- VLM Agents Generate Their Own Memories: Distilling Experience into Embodied Programs (NeurIPS 2024 Spotlight)
  - 检索方式: 向量相似度. 数据库中的demonstration包括input text instruction, textual state, visual state三个部分. 文字部分使用text-embeddingada-002 model计算相似度, 图像部分使用CLIP ViT-B/32 model计算相似度.
  - 检索目的: 将过去任务中agent的轨迹 (即demonstration) 加在prompt中, 帮助agent执行当前任务. 数据库中的demonstration是经过人工修改 (需要人类反馈的参与) 的.
  - Query: 当前的input text instruction, 以及agent的textual state, visual state, 三个部分的embedding.
  - Key: 数据库中存储的input text instruction, 以及agent的textual state, visual state, 三个部分的embedding.
  - Value: Agent的轨迹.
- Voyager: An Open-Ended Embodied Agent with Large Language Models (arxiv 2023)
  - 检索方式: 向量相似度.
  - 检索目的: 通过从skill library中检索出skill, 和环境进行交互, 解决新的任务.
  - Query: 由GPT-3.5生成的解决问题 (任务) 的方案以及environment feedback对应的embedding.
  - Key: 由GPT-3.5生成的可执行代码的描述对应的embedding.
  - Value: 由GPT-4生成的可执行代码 (技能), 这些代码用于agent和环境进行交互.
- JARVIS-1: Open-World Multi-task Agents with Memory-Augmented Multimodal Language Models (arxiv 2023): 将之前的计划存储在multimodal memory中, 在执行新任务时进行召回.
  - 检索方式: 向量相似度. 首先通过query的text embedding和memory中task key的相似度, 选择多个相似度较高的memory entries做为备选. 之后使用CLIP, 计算query的visual embedding和memory中的state key的相似度. 最后选择top-k candidate entries做为prompt的key.
  - 检索目的: 增强agent的规划能力. Memory是多模态的, 每一个entry包括task, plan, state三个部分. 其中plan指的是依次要制造什么物体, state有屏幕截图组成.
  - Query: task描述, 当前agent的state的截图.
  - Key: task描述, state的截图.
  - Value: 文章中将整个memory entry加到prompt中, 具体如何添加不太清楚.
- Ghost in the Minecraft: Generally Capable Agents for Open-World Environments via Large Language Models with Text-based Knowledge and Memory (arxiv 2023)
  - 检索方式: 向量相似度.
  - 检索目的: 帮助当前的计划生成. 在每一个游戏回合中, 当目标达成时, GITM会将完整的计划存放在memory中. 之后通过一个用于总结的LLM, 从多个用于解决同一个目标的计划中总结出关键步骤. 当GITM遇到相似的目标时, 其会从memory中检索出关键步骤, 用于新计划的生成
  - Query: 当前的任务目标对应的embedding.
  - Key: 关键步骤对应的embedding.
  - Value: 相似目标的关键步骤.
- Cradle: Empowering Foundation Agents Towards General Computer Control (NeurIPS 2024)
  - 检索方式: 向量相似度.
  - 检索目的: 文章将其memory分为两类, episodic memory和procedural memory. 综述里面都视为episodic memory.
    - Episodic Memory: 用于维护当前和过去的经验, 其包括输入LMM的视频的关键截图, LMM输出的有用信息, 例如文本和视觉信息, 动作, 任务, 来自各个模块的推理结果. Episodic Memory没有显式的检索过程, 似乎是一段不断维护的prompt.
    - Procedural Memory: Procedural memory专门用于以代码的形式存储和检索技能. 通过检索, 可以更好地进行行动规划. 具体的query, key, value如下.
  - Query: 当前任务的embedding.
  - Key: 技能文档的embedding. 
  - Value: 技能.
- Synapse: Trajectory-as-Exemplar Prompting with Memory for Computer Control (ICLR 2024): 从训练集中选取agent的轨迹加到memory中.
  - 检索方式: 向量相似度.
  - 检索目的: 将之前解决任务的完整轨迹加在prompt中, 帮助LLM (web agent) 解决当前任务.
  - Query: 历史任务的元信息 (metadata, e.g., task descriptions and initial states.) 对应的embedding.
  - Key: 任务的元信息 (metadata, e.g., task descriptions and initial states.) 对应的embedding.
  - Value: 经过化简的, 之前LLM解决问题的轨迹.
- MemoryBank: Enhancing Large Language Models with Long-Term Memory (AAAI 2024)
  - 检索方式: 向量相似度.
  - 检索目的: 通过检索的记忆, 帮助LLM在和用户的沟通过程中不断发展对上下文的理解, 适应用户的个性, 提高LLM在长期互动场景中的表现.
  - Query: 当前对话的上下文对应的embedding.
  - Key: Memory piece对应的embedding.
  - Value: Memory piece, 包括对话轮次, 事件总结等.
- Lift Yourself Up: Retrieval-augmented Text Generation with Self Memory (NeurIPS 2023)
  - 检索方式: 向量相似度.
  - 检索目的: 文章主要解决的是机器翻译的任务. 文章通过迭代地构造用于检索的记忆池, 来使得被检索的样本的分布额和需要解决的query的分布尽可能地接近, 从而提升下游任务的性能. 记忆池中的内容为上一次迭代中LM生成的内容. 通过RAG的方式来构造更好的用于RAG数据库.
  - Query: 需要被解决的input的embedding.
  - Key: text pair中的input部分的embedding.
  - Value: 对应的text pair的output部分的embedding. Text pair中的output部分来自上一轮迭代中LM的输出.
- Tree of Clarifications: Answering Ambiguous Questions with Retrieval-Augmented Large Language Models (EMNLP 2023)
  - 检索方式: 向量相似度, Bing搜索引擎.
  - 检索目的: 文章解决的任务为ASQA (Answer Summaries for Questions which are Ambiguous), 这个任务需要通过外部知识对用户提出的指代不明确的问题 (ambiguous question, AQ) 进行明确 (消除语义模糊的部分, 从而得到DQ, disambiguated question) 以及回答. 文章通过树搜索的方式, 首先生成当前的AQ的多个DQ, 之后继续迭代地为每一个DQ (此时被视为AQ) 生成新的DQ. 最后总结树搜索过程中的所有DQ, 给出原始AQ的全面的答案.
  - Query: question的embedding.
  - Key: 数据库中文章的embedding.
  - Value: 数据库中的文章.
- Retrieval-Generation Synergy Augmented Large Language Models (arxiv  2023)
  - 检索方式: 向量相似度
  - 检索目的: 利用外部知识来完成QA任务. 文章通过迭代地使用RAG (Retrieval Augmented Generation) 和GAR (Generation Augmented Retrieval) 来生成新的query, 从数据库中召回新的文本, 从而逐渐生成更加准确的回答.
  - Query: 用户提出的query ($q$) 以及模型上一轮迭代生成的答案 ($y_{t-1}$) 拼接而成的字符串 ($q_t=[q;y_{t-1}]$) 对应的embedding.
  - Key: 数据库中问本对应的embedding.
  - Value: 数据库中的文本.


## Reasoning Action & Planning Action

侧重reasoning和planning, 侧重目的. Memory如何帮助到plan和reason. 

在同一个任务中 (intra episodic), working memory中的思维轨迹, 动作轨迹, 环境的观察轨迹, 可以帮助LLM一步一步地完成一个任务. LLM能够总结存储在memory中的, 之前步骤中的经验或者错误, 最终完成任务. 推理行为和working memory相互结合后, 可以增强LLM-based agent完成任务的性能.

经过特殊设计的推理策略 (如Tree of thought) 能够进一步提高LLM的推理能力, 帮助LLM适应到他在预训练过程中没有见过的任务 (24点).

Inter episodic相当于是在intra episodic的基础上更上一层楼.

Test time adaption (inference time adaption).

> [!NOTE]
>
> 将相似的文章放到一起.
>
> 可能的tag:
>
> - Tree of thought
> - ReAct: 基本等于利用了当前轨迹的错误经验的文章. (在线推理)
> - Chain of thought: 如果一篇文章被归为Tree of thought或者ReAct, 则其不会再被归为Chain of thought.
> - Reflexion: 如果有利用之前完整的episodic的错误经验, 则为Reflexion. Reflexion一定出现在同一个任务的不同trials中. (在线推理)
> - MCTS (Monte Carlo Tree Search)
> - Multi-agent
> - Task decomposition
>
> Tag字段最后不加句号, 方便复制.
>
> 如果一篇文章之前没有出现过, 则还会有 "其他" 字段.

### Intra Episodic

- Tree of Thoughts: Deliberate Problem Solving with Large Language Models (NeurIPS 2023)
  - 目的: 通过树结构提高了LLM的规划能力.
  - 特点: 提出了Tree of Thought.
  - Tag: ReAct, Reflexion, Multi-agent.
  - 其他: 提出了Tree of Thought.

- ReAct: Synergizing Reasoning and Acting in Language Models (ICLR2023)
  - 目的: 通过prompt让LLM在推理过程中交替生成reasoning traces和actions, 从而提高其生成的结果的准确率. 也可以理解为提高plannin能力.
  - 特点: 提出了ReAct.
  - Tag:  ReAct
  - 其他: 提出了ReAct.

- Reflexion: Language Agents with Verbal Reinforcement Learning (NeurIPS 2023)
  - 目的: 通过加入在之前的尝试中LLM执行相同任务后收到的反馈, 来提高LLM在本次推理过程中的性能.
  - 特点: 提出了Reflexion.
  - Tag: ReAct, Reflexion
  - 其他: 提出了Reflexion.

- Reasoning with Language Model is Planning with World Model (EMNLP 2023)
  - 目的: 借助MCTS提高LLM的推理和规划能力.
  - 特点: 有使用MCTS. 每一次MCTS生成节点的过程 (selection $\rightarrow$ expansion $\rightarrow$ simulation $\rightarrow$ back-propagation) 都可以被视为一个trail, , 因此也有使用reflexion.
  - Tag: ReAct, MCTS
  - 其他: 不在其他部分.

- Large Language Models as Commonsense Knowledge for Large-Scale Task Planning (NeurIPS 2023)
  - 目的: 借助MCTS提高LLM的推理和规划能力. 
  - 特点: 有使用MCTS. 在进行节点扩展的过程中, 会向LLM询问相关信息. 做为world model的LLM负责维护物品的状态, 做为heuristic policy的LLM负责生成action.
  - Tag: ReAct, MCTS, Multi-agent
  - 其他: 不在其他部分 (241124 论文中加在了retrieval from semantic memory).

- SwiftSage: A Generative Agent with Fast and Slow Thinking for Complex Interactive Tasks (NeurIPS2023 (spotlight))
  - 目的: 通过LLM (GPT4, SAGE模块) 和SLM (T5-large, SWIFT模块) 共同进行规划, 提高LLM的规划能力, 控制开销.
  - 特点: SAGE模块负责更正错误, 重新生成任务 (体现Reflexion), 提出子目标; SWIFT模块负责生成action.
  - Tag: ReAct, Reflexion, Multi-agent.
  - 其他: 不在其他部分 (241124 论文中加在了retrieval from semantic memory).

- ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs (ICLR 2024 spotlight)

  - 目的: 通过DFSDT (depth-first search-based decision tree) 选择合适的工具调用轨迹, 使得LLM通过工具调用完成任务.
  - 特点: 通过树搜索的方式增强了LLM的推理能力, 选择合适的工具.
  - Tag: Tree of Thoughts, ReAct, Reflection

- EASYTOOL: Enhancing LLM-based Agents with Concise Tool Instruction (ICLR 2024 Workshop)

  - 目的: 通过任务分解以及多LLM共同推理, 使得agent通过工具调用完成任务.
  - 特点: 通过任务分解降低每一个子任务的难度, 使用不同prompt模板的LLM共同推理.
  - Tag: ReAct, Multi-agent, Task decomposition

- Small LLMs Are Weak Tool Learners: A Multi-LLM Agent (EMNLP 2024)

  - 目的: 文章将工具调用分解为task planning, tool invocation, result summarization三个部分, 分配给planner, caller, 以及summarizer三个LLM, 通过DFSDT的方式进行工具调用, 完成任务.
  - 特点: 通过树搜索的方式增强了LLM的推理能力, 通过任务分解降低每一个子任务的难度, 负责不同部分的LLM被单独微调.
  - Tag: Tree of Thoughts, ReAct, Reflexion, Multi-agent, Task decomposition
  - 其他: 不属于grounding action.

- Look Before You Leap: Towards Decision-Aware and Generalizable Tool-Usage for Large Language Models (arxiv 2024)

  - 目的: 通过微调LLM, 使得其能够决定 (a) 直接完成任务 (b) 通过工具调用完成任务 (c) 通过搜索数据库或互联网完成任务.
  - 特点: 使用不同prompt模板的LLM负责任务的不同部分 (如决定是否调用工具, 调用什么工具).
  - Tag: Multi-agent, Task decomposition
  - 其他: 侧重于让LLM能够根据需要调用工具, 从而提高推理能力.

- ToolNet: Connecting Large Language Models with Massive Tools via Tool Graph (arxiv 2024)

  - 目的: 使得LLM通过工具调用完成任务.
  - 特点: 通过有向图来管理所有的工具, 在LLM完成一个工具的调用后, 通过有向图引导LLM选择合适的工具.
  - Tag: ReAct
  - 其他: 不属于grounding action.

- API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs (EMNLP 2023)

  - 目的: 通过多个LLM, 根据任务生成工具的调用轨迹, 构造数据集.
  - 特点: 通过多个LLM生成高质量数据集.
  - Tag: ReAct, Multi-agent, Task decomposition

- SteP: Stacked LLM Policies for Web Actions (COLM 2024)

  - 目的: 通过人工进行策略的划分, 帮助LLM更好地完成web任务. 在执行一个策略时, LLM可以调用其他策略.
  - 特点: 每一个策略对应一个prompt模板, 即对应一个agent.
  - Tag: ReAct, Multi-agent (不属于Task decomposition, 因为其只是根据当前要执行的策略选择合适的prompt)

- LASER: LLM Agent with State-Space Exploration for Web Navigation (FMDM@NeurIPS2023)

  - 目的: 将任务建模为state-space exploration, 每一个state定义了LLM的动作. LLM通过在state间的转移最终完成任务.
  - 特点: 每一个状态中LLM的promp模板都不一样.
  - Tag: ReAct, Multi-agent

- ADaPT: As-Needed Decomposition and Planning with Language Models (NAACL 2024 (findings))

  - 目的: 根据子任务的执行结果, 动态地将复杂任务分解为子任务, 避免了一个过于复杂的子任务导致整个任务的失败, 从而完成任务.
  - 特点: 在执行任务的过程中动态地进行任务的分解. 
  - Tag: ReAct, Multi-agent, Task decomposition
  - 其他: 不属于grounding action.

- WebPilot: A Versatile and Autonomous Multi-Agent System for Web Task Execution with Strategic Exploration (arxiv 2024)

  - 目的: 通过任务的分解, 利用之前的反思辅助本次动作的生成, 在web空间中进行搜索.

  - 特点: Reflexion体现在两个部分 (a) WebPilot会在其完成每一个子任务后, 对本次子任务的执行进行反思, 利用反思促进下一次子任务的执行. (b) 通过MCTS选择搜索树中要探索的节点. 不同prompt模板的LLM负责不同的部分 (如任务分解, 决定下一个action). 每一次通过MCTS选择节点都可以视为利用了上一次完整的episodic. 

    不同LLM负责搜索的不同部分, 如任务分解, 生成反思, 生成动作等.

  - Tag: ReAct, Reflexion, MCTS,  Multi-agent, Task decomposition

- Learning Evolving Tools for Large Language Models (arxiv 2024)

  - 目的: 提高LLM在动态环境中的工具调用能力, 能够更新prompt中的工具定义.
  - 特点: 使用self-reflection让LLM根据之前的工具调用结果总结正确的工具调用, 使用MCTS让LLM在环境中探索, 寻找正确的工具调用轨迹.
  - Tag: ReAct, Reflexion, MCTS

- Tree Search for Language Model Agents (submitted to ICLR 2025)

  - 目的: 通过最佳优先树搜索提高web agent的规划能力.
  - 特点: 允许LLM在搜索过程中, 通过树状结构进行明确的搜索和回溯, agent可以识别更好的动作轨迹, 从而避免错误的积累, 或者进入难以恢复的错误状态.
  - Tag: Tree of thought
  - 其他: 不属于grounding action.

- Describe, Explain, Plan and Select: Interactive Planning with Large Language Models Enables Open-World Multi-Task Agents (NeurIPS 2023 poster): 

  - 目的: 通过多个LLM共同进行任务规划, 通过Reflexion提高规划能力.
  - 特点: LLM分别充当planner和explainer, 处理来自环境的反馈. Planner会更具反馈重新进行任务规划 (re-plan), 这体现了Reflexion.
  - Tag: ReAct, Reflexion, Multi-agent

- VillagerAgent: A Graph-Based Multi-Agent Framework for Coordinating Complex Task Dependencies in Minecraft (ACL 2024 findings): 多LLM构造计划, 多agent共同执行计划.

  - 目的: 通过文章提出的多agent框架, 实现任务的分解以及子任务的动态分配, 提高多agent的协同效率.
  - 特点: 由多个LLM进行任务的分解, 控制agent, 管理agent状态和环境状态等. 框架中实际行动的agent会使用ReAct等增强性能.
  - Tag: ReAct, Multi-agent, Task decomposition (将任务分解及子任务分配这件事交给多个LLM)

- Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents (ICML 2022)

  - 目的: 利用LLM生成agent的行动计划.
  - 特点: 假如LLM生成了不可以被环境理解的动作, 则通过Sentence-BERT以及embedding相似度将当前动作转化为环境可以理解的动作.
  - Tag: ReAct

- Tree-Planner: Efficient Close-loop Task Planning with Large Language Models (NeurIPS 2023)

  - 目的: 文章构建一棵action tree来进行task planning, 提高LLM的规划能力.
  - 特点: 会先在不与环境进行交互的情况下进行多次规划, 生成action tree. 之后在action tree的约束下与环境进行交互, 限制agent只从action tree中选择动作.
  - Tag: ReAct
  - 其他: 不在其他部分.


### Inter Episodic

- LLMs in the Imaginarium: Tool Learning through Simulated Trial and Error (ACL 2024): LLM通过设想工具可能的使用场景, 进行工具调用, 得到用于微调或者in-context-learning的工具调用轨迹. 工具调用轨迹还会被用于之后的工具调用场景生成.
  - 目的: 通过LLM设想工具可能的使用场景, 生成用于微调或者in-context-learning的高质量工具调用轨迹.
  - 特点: 利用之前生成的工具使用场景 (即不同的task, 不同的trail) 来生成新的使用场景, 生成质量尽可能高的工具调用轨迹. 利用这些轨迹微调LLM, 可以使得LLM能够更好地掌握如何使用工具.
  - Tag: ReAct
- Large Language Models Are Semi-Parametric Reinforcement Learning Agents (NeurIPS 2023)
  - 目的: 使得LLM能够根据过去的交互经验进行更好的决策. 将来自多个任务的交互经验综合起来进行更好的决策.
  - 特点: 在更新交互经验的 $Q$ 值时, 使用了强化学习的相关公式来更新一个动作在特定任务, 观察下的优劣程度.
  - Tag: ReAct
- VLM Agents Generate Their Own Memories: Distilling Experience into Embodied Programs (NeurIPS 2024 Spotlight): 
  - 目的: 利用VLM本身, 在人类帮助下, 从次优演示中生成能够用于上下文学习的演示. 演示被存储在数据库中, 在VLM进行推理时, 其通过RAG从数据库中检索出最相关的轨迹加在prompt中, 增强VLM的规划能力.
  - 特点: 通过人类来生成更好的演示.
  - Tag: ReAct, Multi-agent (不同的agent负责生成演示的不同部分), Task decomposition (在生成演示时进行任务分解, 而不是执行过程中)
- Voyager: An Open-Ended Embodied Agent with Large Language Models (arxiv 2023)
  - 目的: GPT4生成的自动课程能够帮助LLM不断探索世界, 提高规划能力. 在执行某一个任务的过程中, 可以利用之前任务学习到的技能 (可执行程序), 以及学习新的技能.
  - 特点: 用之前任务的技能解决当前任务. 利用其他LLM来校验当前生成的可执行程序的准确性.
  - Tag: ReAct, Multi-agent (有的LLM负责生成技能, 有的LLM负责校验技能, 文章并没有说将任务进行分解)
- See and Think: Embodied Agent in Virtual Environment (ECCV 2024): 多LLM共同生成计划. 文中将使用不同prompt模板的LLM定义为不同的agent.
  - 目的: 通过多LLM共同生成计划, 提高agent的规划能力. 文章直接将使用不同prompt模板的LLM定义为不同的agent. 通过课程学习中的任务帮助agent了解环境, 之前任务中的经验会被用于之后的任务 (变成prompt加在输入中).
  - 特点: 使用课程学习与环境互动. 通过Chain of Summarization method提炼经验, 防止经验太多.
  - Tag: ReAct, Multi-agent, Task decomposition
- Ghost in the Minecraft: Generally Capable Agents for Open-World Environments via Large Language Models with Text-based Knowledge and Memory (arxiv 2023)
  - 目的: 通过不同的LLM进行任务的分解, 计划生成, 动作执行等步骤, 提高LLM的规划能力. 利用其他任务的经验来解决当前任务.
  - 特点: 当目标达成时, GITM会将完整的计划存放在memory中. LLM会对同一个目标对应的多个计划进行总结, 用于之后相似目标的计划生成.
  - Tag: ReAct, Reflexion, Multi-agent, Task decomposition
- Cradle: Empowering Foundation Agents Towards General Computer Control (NeurIPS 2024): 多LMM共同生成计划.
  - 目的: VLM通过Episodic Memory, Procedural Memory来总结过去任务的经验, 进行当前的任务规划.
  - 特点: 利用Episodic Memory维护当前和过去的经验. 会定期对最近添加的多模态信息抽象为长期摘要 (文字形式), 便于检索和存储. 利用Procedural Memory存储之前生成的代码.
  - Tag: ReAct, Reflexion, Multi-agent, Task decomposition
- Planning with Large Language Models via Corrective Re-prompting (FMDM@NeurIPS2022)
  - 目的: 通过LLM生成embodied agent的计划.
  - 特点: 在构造prompt时, 文章会从demonstration set中挑选出和当前需要解决的任务最相关的任务的计划加在prompt中. 当LLM生成的某一个步骤不可执行时, 将来自环境的反馈 (precondition errors) 加到prompt中, 使得其生成正确的步骤.
  - Tag: ReAct

# 241119

终身学习是一个场景, 不是一种技术.

Prioritized Experience Replay提出了回放策略.

# 241126

- Learning Evolving Tools for Large Language Models (arxiv2024): 工具学习strongly related, 还没加到文章中.
