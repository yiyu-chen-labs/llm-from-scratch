# ===== 第一塊：State =====
from typing import TypedDict


class AgentState(TypedDict):
    query: str        # 使用者問題（輸入）
    plan: str         # plan 節點決定的工具名
    result: str       # act 節點執行的結果
    attempts: int     # 試了幾次（煞車用）
    ok: bool          # check 判斷結果好不好


# ===== 第二塊：節點函式 =====
def plan_node(state: AgentState) -> dict:
    """看 query 決定要哪個工具，只回傳要更新的欄位。"""
    query = state["query"]
    if "+" in query:
        decision = "add"
    elif "weather" in query.lower() or "天気" in query:
        decision = "get_weather"
    else:
        decision = "none"
    print(f"[plan]  query={query!r} -> decision={decision}")
    return {"plan": decision}


def act_node(state: AgentState) -> dict:
    """依 plan 執行（先用假的、不花錢）。只回傳要更新的欄位。"""
    plan = state["plan"]
    if plan == "add":
        result = "fake add result: 8"
    elif plan == "get_weather":
        result = "fake weather: 22C sunny"
    else:
        result = "no tool matched"
    print(f"[act]   plan={plan} -> result={result}")
    return {"result": result}


def check_node(state: AgentState) -> dict:
    """看 act 的結果好不好，並記錄試了幾次。"""
    result = state["result"]
    attempts = state["attempts"] + 1
    ok = "fake" in result and "no tool" not in result
    print(f"[check] attempt={attempts} result={result!r} -> ok={ok}")
    return {"ok": ok, "attempts": attempts}


# ===== 第三塊：決定流向的判斷器（給條件邊用）=====
MAX_ATTEMPTS = 3


def decide_next(state: AgentState) -> str:
    """讀 state，回傳下一步去哪。這就是 break 的替代品。"""
    if state["ok"]:
        print("[route] ok=True -> END")
        return "end"
    if state["attempts"] >= MAX_ATTEMPTS:
        print(f"[route] 試了 {state['attempts']} 次還不行 -> END（放棄）")
        return "end"
    print(f"[route] ok=False, attempts={state['attempts']} -> 繞回 plan")
    return "retry"


# ===== 第四塊：畫圖 + 執行 =====
from langgraph.graph import StateGraph, START, END

builder = StateGraph(AgentState)
builder.add_node("plan", plan_node)
builder.add_node("act", act_node)
builder.add_node("check", check_node)

builder.add_edge(START, "plan")
builder.add_edge("plan", "act")
builder.add_edge("act", "check")          # act 做完 -> check（不再直接 END）

builder.add_conditional_edges(
    "check",
    decide_next,
    {
        "retry": "plan",   # 繞回去（環！）
        "end": END,        # 結束
    },
)

graph = builder.compile()


if __name__ == "__main__":
    """frist attempt"""
    #result = graph.invoke({"query": "5 + 3", "attempts": 0})
    """second attempt"""
    result = graph.invoke({"query": "tell me a joke", "attempts": 0})
    print("\nfinal state:", result)