<!-- AI应用开发方向面试题库 -->
<!-- 每个 ## 块是一个独立检索单元（chunk） -->

## RAG系统设计 - 生产级高可用架构
标签: RAG, 向量数据库, 系统设计, 高可用, 检索增强生成

### 面试问题
如何设计一个生产级别的高可用 RAG 系统？请描述核心组件和关键设计决策。

### 考察要点
- 向量数据库选型（Pinecone/Weaviate/Qdrant/ChromaDB）及各自权衡
- 检索策略：dense（稠密向量）、sparse（BM25 稀疏）、hybrid（混合）
- 上下文窗口管理：文档截断、滑动窗口、层级摘要
- 文档分块策略（固定大小 vs 语义感知分块）
- 失败回退：向量库宕机时的降级方案
- 评估指标：recall@k、MRR、NDCG、faithfulness

### 追问方向
- 如果向量数据库查不到结果，有什么保底方案？
- 如何评估 RAG 系统的检索质量？用什么指标？
- 文档超出 LLM 上下文窗口时如何处理？
- Hybrid retrieval 中 dense 和 sparse 权重如何调优？

---

## RAG系统设计 - 文档分块与索引策略
标签: RAG, chunking, 分块, 索引, 嵌入向量, embedding

### 面试问题
RAG 系统中文档分块策略会显著影响检索质量。你会如何设计分块方案？

### 考察要点
- 固定大小分块 vs 语义分块 vs 基于段落/标题的分块
- chunk_size 和 chunk_overlap 的选择依据
- Parent-Child 分块：小块检索 + 大块上下文
- 多粒度索引：同一文档存多个粒度的 embedding
- 元数据附加：标题、来源、页码、创建时间

### 追问方向
- chunk_size 设太大或太小各有什么问题？
- 对 PDF、代码、表格等异构文档分别怎么处理？
- 如何处理跨 chunk 的语义断裂问题（内容被截断）？

---

## RAG系统设计 - 评估与迭代优化
标签: RAG, 评估, RAGAS, 优化, 质量指标, faithfulness

### 面试问题
如何系统性地评估和改进一个 RAG 系统的效果？

### 考察要点
- 端到端评估 vs 组件级评估（检索质量 vs 生成质量）
- RAGAS 框架：faithfulness、answer relevancy、context recall
- 离线评估（标注数据集）vs 在线评估（用户反馈）
- 检索层优化：调整 top_k、RRF 融合权重、重排序（reranker）
- 生成层优化：prompt 工程、温度参数、上下文压缩

### 追问方向
- 如果没有标注数据集，怎么做初步评估？
- 如何构建一个 RAG 评估的黄金测试集？
- Faithfulness 低说明什么问题？如何修复？

---

## RAG系统设计 - 多路召回与重排序
标签: RAG, hybrid search, reranker, 重排序, BM25, 召回

### 面试问题
请描述 Hybrid Search + Reranker 的完整检索流程，以及为什么这比单纯向量检索效果好。

### 考察要点
- Dense retrieval（ANN 近似最近邻）的优势与局限（精确关键词匹配弱）
- Sparse retrieval（BM25/TF-IDF）的优势与局限（语义理解弱）
- Hybrid 融合方法：RRF（Reciprocal Rank Fusion）、加权线性融合
- Reranker 作用：Cross-encoder 精排，比 bi-encoder 更精准但更慢
- 延迟与质量的权衡

### 追问方向
- RRF 中的 k 参数（通常为 60）有什么意义？如何调优？
- 为什么 bi-encoder 检索快但精度较低？
- 什么场景下不需要 reranker？

---

## LLM应用架构 - Token管理与成本优化
标签: LLM, token, 成本优化, 上下文压缩, prompt优化

### 面试问题
在构建 LLM 应用时，token 成本是重要考虑因素。你会如何优化 token 使用？

### 考察要点
- Prompt 压缩：去除冗余、保留关键信息
- 上下文缓存（KV-cache）：相同前缀命中率
- 对话历史截断策略：滑动窗口、摘要压缩
- 模型降级：简单查询用小模型，复杂任务用大模型
- 输出 token 控制：max_tokens 设置、structured output

### 追问方向
- 如何在不损失太多质量的前提下减少 50% 的 token？
- 对话多轮后上下文过长怎么处理？
- 如何监控线上应用的 token 消耗？

---

## LLM应用架构 - 流式输出与响应速度
标签: LLM, 流式输出, streaming, 延迟, 用户体验, async

### 面试问题
如何优化 LLM 应用的响应速度和用户体验？描述你会采取的技术手段。

### 考察要点
- Streaming 输出：SSE（Server-Sent Events）/ WebSocket 实时推送
- 并发处理：asyncio、并行调用多个 LLM
- 预填充（Prefill）缓存：system prompt 缓存
- TTFT（Time To First Token）vs 总延迟的权衡
- 骨架显示（Skeleton UI）改善体验感知

### 追问方向
- Streaming 在后端如何实现？前端如何消费？
- 如何测量 LLM API 的 P50/P95 延迟？
- 如果用户网络中断，流式输出怎么处理？

---

## LLM应用架构 - 错误处理与可靠性
标签: LLM, 错误处理, 重试, fallback, 可靠性, 幂等

### 面试问题
LLM API 调用存在超时、限流、幻觉等问题。如何设计一个健壮的 LLM 应用？

### 考察要点
- 重试策略：指数退避、最大重试次数、抖动（jitter）
- 限流处理：令牌桶算法、队列缓冲
- 输出验证：JSON Schema 校验、幻觉检测
- 回退策略：降级到缓存结果或规则引擎
- 超时设置：connect_timeout vs read_timeout

### 追问方向
- LLM 返回了格式错误的 JSON，如何优雅处理？
- 如何检测 LLM 的幻觉输出？
- 多个 LLM provider（OpenAI + Claude + Gemini）如何做 failover？

---

## Agent与工具调用 - 设计模式
标签: Agent, 工具调用, tool use, function calling, ReAct, Plan-and-Execute

### 面试问题
请描述你了解的 LLM Agent 设计模式，以及各自适合的场景。

### 考察要点
- ReAct（Reason + Act）：循环推理-行动，适合探索性任务
- Plan-and-Execute：先规划再执行，适合确定性强的工作流
- Tool Use / Function Calling：结构化工具调用接口设计
- Multi-Agent：专业化 agent 协作，Supervisor 路由
- 工具定义原则：单一职责、清晰描述、参数类型明确

### 追问方向
- ReAct 的最大问题是什么？（循环过长、容易陷入死循环）
- 如何限制 Agent 的执行步数避免无限循环？
- Multi-Agent 中 agent 间如何传递上下文？

---

## Agent与工具调用 - MCP协议与工具生态
标签: MCP, Model Context Protocol, 工具生态, stdio, agno

### 面试问题
什么是 MCP（Model Context Protocol）？它解决了什么问题？你是否有实际使用经验？

### 考察要点
- MCP 是 Anthropic 提出的开放协议，统一 LLM 与外部工具/数据源的接口
- 传输层：stdio（本地子进程）vs HTTP/SSE（远程服务）
- 工具注册与发现：server 声明 tools，client 动态调用
- 相比直接 API 调用的优势：标准化、可复用、跨框架
- 与 agno、Claude、OpenAI 等框架的集成方式

### 追问方向
- MCP server 和普通 REST API 有什么本质区别？
- stdio 传输为什么适合本地工具？有什么限制？
- 如何在 MCP server 中处理长时间运行的操作？

---

## Agent与工具调用 - 状态管理与记忆
标签: Agent, 记忆, memory, 状态管理, 对话历史, 持久化

### 面试问题
如何为 LLM Agent 设计记忆系统？区分短期记忆和长期记忆。

### 考察要点
- 短期记忆：对话上下文窗口内的历史（in-context）
- 长期记忆：外部存储（数据库、向量库）持久化用户偏好/事实
- 记忆写入策略：何时写、写什么、如何避免噪声
- 记忆检索：语义相似度匹配历史相关片段
- 隐私与安全：多用户隔离、敏感信息过滤

### 追问方向
- 对话历史保存多长？如何压缩长历史？
- 如何让 Agent 记住"上次用户说他喜欢简洁的回答"？
- 多用户系统中如何隔离各自的记忆空间？

---

## 向量数据库与检索 - 选型与权衡
标签: 向量数据库, ChromaDB, Pinecone, Qdrant, Weaviate, 选型

### 面试问题
常见向量数据库（ChromaDB、Pinecone、Qdrant、Weaviate）各有什么特点？你会如何选型？

### 考察要点
- ChromaDB：本地/嵌入式，适合开发原型，无需部署
- Pinecone：全托管 SaaS，开箱即用，成本高
- Qdrant：开源，支持 payload 过滤，性能优秀，可自托管
- Weaviate：开源，内置多种检索模式，schema 较复杂
- 选型维度：规模、延迟要求、运维成本、过滤能力

### 追问方向
- 数据量从 100 万增长到 10 亿向量，架构如何演进？
- 向量数据库的 payload 过滤（metadata filter）如何影响性能？
- 为什么不直接用 PostgreSQL + pgvector？

---

## 向量数据库与检索 - ANN算法原理
标签: ANN, HNSW, IVF, 近似最近邻, 向量检索原理

### 面试问题
向量数据库中的近似最近邻（ANN）算法是如何工作的？请描述 HNSW 的基本原理。

### 考察要点
- 暴力搜索（O(N)）的问题：数据量大时不可接受
- HNSW（Hierarchical Navigable Small World）：分层图结构，从高层图粗定位再到低层精确匹配
- IVF（Inverted File Index）：聚类后只搜索最近的几个簇
- 精度（recall）vs 速度（QPS）vs 内存的三角权衡
- ef_construction、M 等参数对 HNSW 的影响

### 追问方向
- HNSW 构建索引的时间复杂度是多少？
- recall@10=0.95 是什么含义？如何测量？
- 什么时候暴力搜索反而是最优选择？

---

## 提示工程 - 结构化输出与JSON强制
标签: 提示工程, prompt engineering, JSON, 结构化输出, function calling

### 面试问题
如何可靠地让 LLM 输出结构化 JSON？描述你用过的几种方法及各自的优缺点。

### 考察要点
- Prompt 约束：直接在 prompt 中要求 JSON，并给出示例（few-shot）
- Function Calling / Tool Use：OpenAI/Anthropic 内置的结构化输出机制
- JSON Mode / Structured Outputs API：强制输出合法 JSON
- 后处理兜底：正则提取 + 格式修复
- Pydantic 集成：schema 自动生成 prompt + 输出校验

### 追问方向
- 如果 LLM 输出的 JSON 缺少某个字段怎么处理？
- Function Calling 和 JSON Mode 有什么区别？
- 为什么即使用了 JSON Mode，有时输出仍然不合法？

---

## 提示工程 - 思维链与推理增强
标签: CoT, 思维链, Chain-of-Thought, 推理, few-shot

### 面试问题
什么是思维链（Chain-of-Thought）提示？在什么情况下使用它？有哪些变体？

### 考察要点
- CoT：让模型先输出推理步骤再给出答案，提升复杂任务准确率
- Zero-shot CoT："Let's think step by step" 简单触发
- Few-shot CoT：提供带推理过程的示例
- Self-Consistency：多次采样取多数票（提升稳定性）
- Tree-of-Thought（ToT）：多路径探索，适合需要回溯的问题

### 追问方向
- CoT 什么时候反而会降低性能？（简单任务、过长推理链）
- 如何在保持推理质量的同时减少 CoT 的 token 消耗？
- Self-Consistency 中怎么合并多个不一致的答案？

---

## 模型部署与优化 - vLLM与推理加速
标签: vLLM, 推理加速, 部署, PagedAttention, 吞吐量, GPU

### 面试问题
请描述 vLLM 的核心优化原理，以及在生产部署中需要关注的关键参数。

### 考察要点
- PagedAttention：KV-cache 分页管理，减少内存碎片，提升并发
- Continuous Batching：动态批处理，让短请求不等长请求
- Tensor Parallelism：多 GPU 张量并行，适合大模型
- 关键参数：max_model_len、max_num_seqs、gpu_memory_utilization
- Tool calling 支持：--enable-auto-tool-choice、--tool-call-parser

### 追问方向
- PagedAttention 和普通 KV-cache 有什么本质区别？
- 如果 GPU 内存不足，有哪些降低显存的方法？
- vLLM 的 tensor parallel size 如何决定？

---

## 模型部署与优化 - 量化与模型压缩
标签: 量化, INT8, GPTQ, AWQ, LoRA, 模型压缩, 微调

### 面试问题
什么是模型量化？常见的量化方法有哪些？量化会带来什么影响？

### 考察要点
- 量化原理：将 FP32/FP16 权重压缩为 INT8/INT4，减少显存占用
- PTQ（训练后量化）：GPTQ、AWQ，精度损失小，无需重训
- QAT（量化感知训练）：训练时模拟量化，精度最好但成本高
- LoRA 微调：低秩适配，只训练少量参数，冻结主模型
- 质量权衡：INT4 比 INT8 更省显存但精度损失更大

### 追问方向
- AWQ 比 GPTQ 好在哪里？
- LoRA 的 rank 参数如何选择？太大太小各有什么影响？
- 量化后如何快速验证模型质量没有大幅下降？

---

## 模型部署与优化 - API服务设计
标签: API设计, 负载均衡, 限流, 监控, OpenAI兼容, 生产部署

### 面试问题
如何设计一个生产级别的 LLM API 服务？需要考虑哪些非功能性需求？

### 考察要点
- OpenAI 兼容接口：/v1/chat/completions 标准，方便客户端切换
- 负载均衡：多实例 LLM，nginx/Envoy 上游路由
- 限流与配额：按用户/API Key 的 RPM、TPM 限制
- 监控告警：QPS、延迟分位数（P50/P99）、错误率
- 健康检查与优雅重启

### 追问方向
- 如何在不停机的情况下更新 LLM 模型版本？
- 如果某个请求卡住了（LLM 超时不返回），如何处理？
- 如何追踪每个请求的 token 消耗和费用？

---

## 项目经验 - RAG系统项目描述
标签: 项目经验, RAG, 模块化设计, MCP, 系统架构, 简历

### 面试问题
请介绍你做过的最有挑战性的 AI 应用项目，重点说明架构设计和遇到的核心问题。

### 考察要点（STAR 框架）
- Situation：项目背景、规模、用户场景
- Task：你负责的具体模块和目标
- Action：关键技术决策，为什么这样设计
- Result：量化的结果（准确率、延迟、用户数等）

### 评分锚点
- 优秀回答：有具体数据（"检索召回率从 62% 提升到 81%"），能描述技术权衡
- 良好回答：有具体技术选型，能说清楚为什么
- 及格回答：能描述项目做了什么，但没有数据或深度

### 追问方向
- 这个项目中你最大的技术挑战是什么？怎么解决的？
- 如果重来，你会做哪些不同的设计决策？
- 项目上线后遇到过什么生产问题？如何排查的？

---

## 项目经验 - 技术选型决策
标签: 技术选型, 权衡, 架构决策, 为什么选择

### 面试问题
在你的项目中，你是如何做技术选型决策的？请以一个具体的选型为例说明。

### 考察要点
- 明确需求约束（性能、成本、团队熟悉度、维护成本）
- 列举候选方案的优缺点对比
- 用数据或 POC 验证选型
- 决策的可撤销性（避免过早锁定）

### 评分锚点
- 优秀：能说出"当时 A 比 B 好因为我们实测了 X，但如果规模更大应该选 C"
- 良好：有明确的对比维度
- 及格：只说"我们选了 A 因为大家都用"

### 追问方向
- 你考虑过哪些备选方案？最终为什么没选？
- 这个决策后来被证明是正确的吗？有没有让你后悔的地方？

---

## 行为题 - 技术难题与攻坚
标签: 行为题, STAR, 问题解决, 技术难题, 攻坚

### 面试问题
描述一次你遇到了很难的技术难题，你是怎么解决的？

### 考察要点（STAR）
- Situation：什么背景下遇到的问题
- Task：问题的严重程度和你的职责
- Action：调试过程——如何排查、用了什么工具、如何缩小范围
- Result：最终解决方案，以及从中学到了什么

### 评分锚点
- 优秀：有系统的调试思路，能量化影响，有复盘和改进措施
- 良好：有清晰的问题描述和解决过程
- 及格：能描述大致过程但缺乏细节

### 追问方向
- 当时有没有考虑过其他解决方案？为什么没选？
- 这个问题如何避免在未来再次发生？
- 在解决这个问题过程中，有没有向别人寻求帮助？

---

## 行为题 - 学习新技术
标签: 行为题, 学习能力, 自驱力, 新技术, 成长

### 面试问题
AI 领域技术更新很快。你是如何保持技术前沿的？举例说明你快速学习一项新技术的经历。

### 考察要点
- 学习资源选择（论文、GitHub、官方文档 vs 二手教程）
- 学习方式：从源码学 vs 从实践学
- 如何快速判断一项技术的价值（避免追新而非追实用）
- 学以致用：有没有把新技术应用到实际项目

### 评分锚点
- 优秀：有具体技术和时间节点，能说出学习路径，并在项目中落地
- 良好：能举出具体例子
- 及格：回答很笼统，比如"我经常看论文"但说不出具体内容

### 追问方向
- 最近六个月你学了什么新技术？它解决了什么问题？
- 你如何区分哪些新技术值得深入学习，哪些只需要了解？
