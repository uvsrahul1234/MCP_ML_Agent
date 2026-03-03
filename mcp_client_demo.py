# mcp_client_demo.py
from __future__ import annotations

import asyncio
import json

from agents.mcp import MCPServerStdio

params = {"command": "uv", "args": ["run", "mcp_ml_server.py"]}


def _extract_text(call_result) -> str:
    """
    CallToolResult is not a list. It typically has `.content`,
    which is a list of content blocks (TextContent, etc.).
    This extracts text safely.
    """
    # Most implementations: call_result.content -> list[TextContent]
    content = getattr(call_result, "content", None)

    if content is None:
        # fallback: some versions store it under .result or similar
        content = getattr(call_result, "result", None)

    if content is None:
        return str(call_result)

    # If it's a list of content blocks
    if isinstance(content, list) and len(content) > 0:
        first = content[0]
        text = getattr(first, "text", None)
        return text if text is not None else str(first)

    return str(content)


def _pretty_json(s: str) -> str:
    try:
        return json.dumps(json.loads(s), indent=2)
    except Exception:
        return s


async def main():
    async with MCPServerStdio(params=params, client_session_timeout_seconds=60) as server:
        tools = await server.list_tools()
        print(f"Found {len(tools)} tools:")
        for t in tools:
            print(" -", t.name)

        # 1) Train and export joblib
        out = await server.call_tool("train_models", {"request_id": "demo-1"})
        text = _extract_text(out)
        print("\ntrain_models:\n", _pretty_json(text))

        # 2) Evaluate metrics
        out = await server.call_tool("confusion_matrix_metrics", {"request_id": "demo-2"})
        text = _extract_text(out)
        print("\nconfusion_matrix_metrics:\n", _pretty_json(text))

        # 3) Select model
        out = await server.call_tool("select_best_model", {"request_id": "demo-3"})
        text = _extract_text(out)
        print("\nselect_best_model:\n", _pretty_json(text))

        # 4) SHAP explain
        out = await server.call_tool("shap_explain_selected", {"request_id": "demo-4", "local_index": 10})
        text = _extract_text(out)
        print("\nshap_explain_selected:\n", _pretty_json(text))

        # 5) Export parquet audit
        out = await server.call_tool("export_parquet_audit", {"request_id": "demo-5", "filename": "submission_audit.parquet"})
        text = _extract_text(out)
        print("\nexport_parquet_audit:\n", _pretty_json(text))


if __name__ == "__main__":
    asyncio.run(main())