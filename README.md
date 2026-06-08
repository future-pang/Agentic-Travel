# 🗺️ Agentic-Travel (智能文旅 Agent)

> **基于 LangGraph 与 Agent Teams 编排架构的智能文旅助手系统**

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/framework-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Spec](https://img.shields.io/badge/spec-Pydantic%20V2-red.svg)](https://github.com/pydantic/pydantic)

🌍 **智能文旅 Agent** 是一个高可控、低延迟的智能文旅助手系统。项目基于 **LangGraph** 状态图实现了 **Agent Teams (主从特工)** 解耦编排架构，并集成多源外部 API（高德算路、和风气象）与垂直领域 **Skills**。针对文旅长周期场景，系统设计了**四层长期记忆系统**与**五层级联上下文窗口压缩管道**，在防范大模型长对话遗忘与 Token 膨胀的同时，提供了工业级的交互流畅度与执行安全管控。

---

## ✨ 核心技术亮点

### 1. Agent Teams 主从编排架构 (Coordinator-Worker)
* **主从执行解耦**：由主控特工（Coordinator）负责轻量级前台流式交互，将复杂且耗时的原子任务（如路线规划、气象查询）分发给独立沙箱中的子特工（Worker）异步并发执行。
* **解耦通信信道**：子特工执行完毕后通过全局异步消息队列投递状态通知主动唤醒主控，主控在子特工运行期间可以通过指令队列动态注入纠偏指令，或直接强杀超时任务，保障系统强可控性。

### 2. 四层长期记忆系统 (4-Layer Memory System)
* **核心分类与过滤**：划分画像、偏好、即时进度与实时反馈四类记忆，并通过排除清单过滤掉寒暄等无关噪声，控制记忆有价值。
* **常驻索引按需加载**：将 `MEMORY.md` 索引常驻于系统 Prompt 中，而具体记忆细节作为独立 Markdown 文件保存在本地，需要时通过三步协议（扫描、选择、加载）按需动态读取，防止撑爆上下文。
* **API 前缀缓存优化**：利用全局静态 System Message 缓存机制，最大化命中大模型侧的 **Prompt Cache (前缀缓存)**，极大地降低长周期交互下的首字延迟与 Token 费用。

### 3. 五层级联上下文压缩管道 (5-Tier Context Compression)
遵循“从粗到细、从无损到有损、从读时投影到状态重写”的级联防线：
* **第一层 Context Trim**：200K 物理边界兜底硬性截断，保障 API 稳定。
* **第二层 History Snip**：150K 阈值自动触发或模型通过短 ID 标签主动调用，将老消息压缩为 snipped 占位桩。
* **第三层 Micro-Compact**：用户闲置超 60 分钟时（云端缓存失效），自动清理大体积的可重建工具输出，保留最新 5 条。
* **第四层 Context Collapse**：请求前动态拦截，用大模型生成老对话段落摘要，折叠发送，且不破坏底层数据库状态。
* **第五层 Auto-Compact**：终极防线，利用大模型对老历史全量摘要，并向 LangGraph 抛出 `RemoveMessage` 永久重写底层图状态。

### 4. 双层持久化与树形指针历史回放
* **SqliteSaver (底层图状态)**：使用 SQLite 持久化存储 LangGraph 运行快照和状态 Checkpoints，实现多特工跨推理轮次的断点续传。
* **JSONL Transcript (上层日志副本)**：自建会话 Transcript 副本，每条记录携带 `uuid` 和 `parentUuid` 组成树形指针链。在恢复会话时通过叶子节点溯源重构，解决多分支、重试等分支分叉冲突。
* **并发写安全**：采用写队列异步批量刷盘（100ms 间隔）并使用多线程物理锁限制文件追加，防范多 Worker 并发写入冲突。

### 5. 工具生态与 Skills 技能包规范
* **最小特权沙箱**：基于 Markdown 与 YAML 规范化编写 Skills 技能包（SOP 流程控制），为子特工唤醒时动态注入 SOP 指令并过滤绑定最小工具集。
* **PersistingToolNode**：拦截并重定向超大工具输出。当返回值字符数大于 50K 时，自动持久化为文本文件并替换为 preview 预览 JSON，提供基于行范围的安全回读工具（`read_local_file`），并利用前缀验证杜绝路径穿越攻击。

---

## 🏗️ 项目结构

```
Agentic_RAG/
├── configs/                  # 配置文件（agent_config.yaml, settings.py）
├── core/                     # 调度核心（消息队列、任务管理器、工具注册、技能包加载器）
├── server/
│   ├── agent/                # 智能体编排
│   │   ├── compression/      # 五层级联上下文压缩管道实现
│   │   ├── node/             # Coordinator 节点及会话存储挂载
│   │   ├── grapy.py          # LangGraph 状态图编译与 SQLite Checkpointer 注册
│   │   └── state.py          # 全局 AgentState 定义
│   ├── memory/               # 四层长期记忆系统（提取、选择注入、索引管理）
│   ├── rag/                  # 知识库向量检索（Ingestion 与 Retrieval）
│   └── tools/                # 外部原子工具集与编排控制工具（spawn_worker, SnipTool）
├── scripts/                  # 自动化隔离测试套件与数据库清空重置脚本
├── skills/                   # 垂直领域 SOP 技能定义文档 (.md)
├── Data/                     # 原始文旅知识库 Markdown 文档
└── main.py                   # 异步 REPL 命令行开发与执行主干 Harness
```

---

## 🚀 快速开始

### 环境要求

* Python 3.10+
* 数据库: SQLite

### 安装部署

1. **克隆项目**
   ```bash
   git clone https://github.com/liunor/Agentic-Travel.git
   cd Agentic-Travel
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **配置环境变量**
   创建并在项目根目录配置 `.env` 文件，填入您的 API Keys：
   ```env
   volcengine_API_KEY="your_volcengine_key"
   DEEPSEEK_API_KEY="your_deepseek_key"
   QWEATHER_API_KEY="your_weather_key"
   AMAP_API_KEY="your_amap_key"
   ```

4. **初始化知识库向量索引 (RAG Ingest)**
   ```bash
   python main.py ingest
   ```

5. **启动命令行交互会话 (Harness REPL)**
   ```bash
   python main.py chat
   ```
   * REPL 支持内置指令：
     * `/sessions` 列出历史会话
     * `/load <session_id>` 恢复特定会话
     * `/new` 新建干净会话
     * `/clear` 清空当前对话上下文
     * `/exit` 安全保存日志并退出

6. **运行自动化隔离测试套件**
   ```bash
   python scripts/test_snip_compact.py
   python scripts/test_tool_persistence.py
   ```

7. **冷启动一键清空重置**
   ```bash
   python scripts/reset_db.py
   ```

---

## 🔧 配置说明

详细的模型、检索与第三方 API（和风气象、高德导航）配置见 `configs/settings.py`，智能体默认及可覆盖模型优先级见 `configs/agent_config.yaml`。

---

## 📄 开源协议

本项目采用 MIT 协议开源。详见 LICENSE 文件。
