# app/graph.py
from langgraph.graph import StateGraph, END
from .nodes import (
    S,
    classify_node,
    retrieve_node,
    rerank_node,
    propose_fix_node,
    judge_node,
    execute_node,
)

g = StateGraph(S)
g.add_node("classify", classify_node)
g.add_node("retrieve", retrieve_node)
g.add_node("rerank", rerank_node)
g.add_node("propose_fix", propose_fix_node)
g.add_node("judge", judge_node)
g.add_node("execute", execute_node)

g.set_entry_point("classify")
g.add_edge("classify", "retrieve")
g.add_edge("retrieve", "rerank")
g.add_edge("rerank", "propose_fix")
g.add_edge("propose_fix", "judge")

# ✅ Route to execute if PASS, otherwise end the graph
g.add_conditional_edges(
    "judge",
    lambda s: "execute" if s.get("verdict") == "PASS" else "__end__",
    {"execute": "execute", "__end__": END},
)

g.add_edge("execute", END)

app = g.compile()
