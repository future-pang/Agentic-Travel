import os
from langchain_core.messages import SystemMessage
from server.memory.manager import MemoryManager, INDEX_FILENAME, MEMORY_DIR

_CACHED_MEMORY_SYSTEM_MESSAGE = None
_LAST_INDEX_MTIME = 0.0

def get_memory_system_message() -> SystemMessage:
    """
    获取包含长期记忆索引及使用指南的系统提示消息 (SystemMessage)。
    1. 仅当 MEMORY.md 索引文件被修改时才重新生成并读取磁盘。
    2. 否则，直接返回内存中的缓存实例，极大节省 CPU 及 IO。
    
    Returns:
        包含当前记忆索引与大模型记忆操作规范的 SystemMessage
    """
    global _CACHED_MEMORY_SYSTEM_MESSAGE, _LAST_INDEX_MTIME
    
    index_path = os.path.join(MEMORY_DIR, INDEX_FILENAME)
    
    # 1. 检查 MEMORY.md 的最新修改时间
    mtime = 0.0
    if os.path.exists(index_path):
        try:
            mtime = os.path.getmtime(index_path)
        except Exception:
            mtime = 0.0
            
    # 2. 如果缓存存在且文件未被修改，则直接命中内存缓存，极速返回
    if _CACHED_MEMORY_SYSTEM_MESSAGE is not None and mtime == _LAST_INDEX_MTIME:
        return _CACHED_MEMORY_SYSTEM_MESSAGE
        
    # 3. 缓存失效，重新加载并组装系统提示词
    manager = MemoryManager()
    index_content = manager.load_memory_index()
    
    prompt = f"""## 峨眉山旅程记忆系统指南 (Travel Memory System Guide)

你拥有一套自动化的长期旅程记忆系统。记忆的写入和检索均由后台自动完成，你**无需手动读取或修改**任何记忆文件。

### 1. 记忆的核心分类与定义 (Travel Memory Taxonomy):
* **persona（基础画像 - 解决”能不能”问题）**: 同行人员、身体限制条件（有腿伤/膝盖痛、恐高、体能极弱）、年龄区间等不易变的特征。
* **preference（游玩偏好 - 解决”想不想”问题）**: 对人文寺庙或自然风景/猴区的喜好、餐饮习惯、倾向交通工具（索道缆车 vs 徒步）等主观选择。
* **realtime_ctx（即时上下文 - 解决”当前进度”问题）**: 游客在景区中的实时进度及环境。如：当前所在点（雷洞坪、清音阁）、天气状况、携带的装备及行李状态、已游览点等。
* **feedback（实时反馈 - 解决”承受度及体验”问题）**: 游客在行程中由于劳累、寒冷等发出的主观抗议（如：”爬山太累了”、”腿快走断了”）。

### 2. 记忆的排除标准 (What NOT to save):
* 绝对不要记录礼貌性、无关紧要的日常寒暄（如”你好”、”谢谢”、”辛苦了”）。
* 绝对不要记录临时的工具调试参数或转瞬即逝的极短话题。

### 3. 当前已保存的旅程记忆索引 (Travel Memory Index):
以下记忆已由系统自动管理。与当前提问最相关的记忆会在你的上下文中自动注入，你直接使用即可，**绝对不要尝试自己去读取这些文件**。
```markdown
{index_content}
```
"""
    _CACHED_MEMORY_SYSTEM_MESSAGE = SystemMessage(content=prompt)
    _LAST_INDEX_MTIME = mtime
    
    return _CACHED_MEMORY_SYSTEM_MESSAGE
