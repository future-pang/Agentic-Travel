from server.tools.api.weather_tool import (
    get_current_weather,
    get_travel_advice,
    get_astronomy_info,
    get_weather_forecast,
)

from server.tools.api.map_tool import (
    get_walking_plan,
    get_distance,
    search_around,
    get_static_map,
)

from server.tools.api.time_tool import get_current_time
from server.tools.api.web_search_tool import web_search
from server.tools.api.image_generation_tool import generate_image_tool
from server.rag.rag_tool import search_knowledge_base
from server.tools.api.file_read_tool import read_local_file

ALL_AVAILABLE_TOOLS = [
    get_current_weather,
    get_travel_advice,
    get_astronomy_info,
    get_weather_forecast,
    get_walking_plan,
    get_distance,
    search_around,
    get_static_map,
    get_current_time,
    web_search,
    generate_image_tool,
    search_knowledge_base,
    read_local_file,
]

# ── Micro-Compact 可裁剪工具动态注册 ──────────────────────────────────────────
# 规则：「可重新获取」的工具才能注册。
# generate_image_tool 不注册（图片生成不可重复）。
# spawn_worker / send_message / TaskStop / SnipTool 是编排工具，不在此列表中。
def _register_compactable_tools():
    try:
        from server.agent.compression.micro_compact import register_compactable_tool
        _compactable = [
            get_current_weather,
            get_travel_advice,
            get_astronomy_info,
            get_weather_forecast,
            get_walking_plan,
            get_distance,
            search_around,
            get_static_map,
            get_current_time,
            web_search,
            search_knowledge_base,
            read_local_file,
        ]
        for t in _compactable:
            register_compactable_tool(t.name)
    except Exception:
        pass  # micro_compact 模块不可用时静默跳过，不影响主流程

_register_compactable_tools()