"""
峨眉山文旅智能体 记忆系统 - LangChain 专属工具模块。

提供三个工具供 Coordinator 主动调用：
  - save_memory_topic:   保存/更新一条旅程记忆主题文件
  - read_memory_topic:   读取指定记忆主题文件的完整内容
  - search_memory_bm25:  BM25 全文关键词搜索（备用检索工具）

注意：search_memory_bm25 是主动检索的备用手段。
常规流程中，记忆注入由 injection.py 的 LLM 自动选择完成；
当模型需要在记忆目录中进行全文关键词检索时，可主动调用此工具。
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from server.memory.manager import MemoryManager
from server.memory.search import BM25


class DeleteMemoryTopicInput(BaseModel):
    filename: str = Field(
        description="需要删除的记忆文件名，必须以 '.md' 结尾（例如 'traveler_persona.md'）。"
    )


class SaveMemoryTopicInput(BaseModel):
    filename: str = Field(
        description="记忆文件名，必须以 '.md' 结尾（例如 'traveler_persona.md'）。如果属于已有主题，请精确定位同名文件进行覆盖合并。"
    )
    content: str = Field(
        description="记忆的具体详细正文。言简意赅，只记录对后续行程决策起关键性作用的事实。"
    )
    memory_type: str = Field(
        description="记忆分类，必须是 'persona' (画像), 'preference' (偏好), 'realtime_ctx' (即时上下文), 'feedback' (承受度反馈) 之一。"
    )
    description: str = Field(
        description="一句话极简描述该记忆主题，用于显示在 MEMORY.md 索引中。"
    )


class ReadMemoryTopicInput(BaseModel):
    filename: str = Field(
        description="记忆文件名，必须以 '.md' 结尾（例如 'traveler_persona.md'）。"
    )


class SearchMemoryBM25Input(BaseModel):
    query: str = Field(
        description="搜索关键词，支持中英文混合。系统将对所有记忆文件的完整内容进行 BM25 全文检索。"
    )
    limit: int = Field(
        default=5,
        description="返回结果数量上限，默认 5 条。"
    )


@tool("save_memory_topic", args_schema=SaveMemoryTopicInput)
def save_memory_topic(filename: str, content: str, memory_type: str, description: str) -> str:
    """
    保存或更新一个峨眉山长期旅程记忆主题文件。
    该工具可被 Coordinator 用于前台决策时主动记录并持久化重要的游客画像、特定偏好或打卡点，写入后系统会自动重建 MEMORY.md 索引。
    """
    try:
        manager = MemoryManager()
        msg = manager.save_memory_topic(filename, content, memory_type, description)
        return msg
    except Exception as e:
        return f"保存记忆文件失败: {str(e)}"


@tool("read_memory_topic", args_schema=ReadMemoryTopicInput)
def read_memory_topic(filename: str) -> str:
    """
    读取指定记忆主题文件的全部详细内容。
    当 MEMORY.md 索引中有历史记录，且你需要获取其完整、具体的上下文偏好或体能限制细节时调用。
    """
    try:
        manager = MemoryManager()
        content = manager.read_memory_topic(filename)
        return content
    except Exception as e:
        return f"读取记忆文件失败: {str(e)}"


@tool("delete_memory_topic", args_schema=DeleteMemoryTopicInput)
def delete_memory_topic(filename: str) -> str:
    """
    删除指定的记忆主题文件。
    当用户明确要求忘记某项信息，或某个记忆已过时、错误需要清理时调用。
    删除后会自动重建 MEMORY.md 索引。
    """
    try:
        manager = MemoryManager()
        msg = manager.delete_memory_topic(filename)
        return msg
    except Exception as e:
        return f"删除记忆文件失败: {str(e)}"


@tool("search_memory_bm25", args_schema=SearchMemoryBM25Input)
def search_memory_bm25(query: str, limit: int = 5) -> str:
    """
    对记忆目录所有文件进行 BM25 全文关键词检索，返回最相关的记忆条目。
    这是备用检索工具：当你需要在记忆目录中主动搜索某个关键词（如景点名、症状、特定偏好），
    而系统注入的上下文记忆未能覆盖时，可主动调用此工具进行全文搜索。
    注意：此工具会读取所有记忆文件的完整内容，调用频率请适度控制。
    """
    try:
        manager = MemoryManager()
        # 读取全量内容（BM25 需要完整正文进行评分）
        all_memories = manager.scan_memory_full()
        if not all_memories:
            return "记忆目录当前为空，没有可检索的内容。"

        bm25_engine = BM25(all_memories)
        ranked = bm25_engine.rank(query)

        if not ranked:
            return f"BM25 全文检索未找到与「{query}」相关的记忆条目。"

        results = []
        for mem, score in ranked[:limit]:
            results.append({
                "filename": mem["filename"],
                "type": mem["type"],
                "description": mem["description"],
                "score": round(score, 4),
                "content_preview": mem["content"][:300] + ("..." if len(mem["content"]) > 300 else ""),
            })

        import json
        return json.dumps({
            "query": query,
            "total_found": len(ranked),
            "results": results,
            "tip": "如需查看某条记忆的完整内容，请调用 read_memory_topic 工具。"
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return f"BM25 全文检索失败: {str(e)}"
