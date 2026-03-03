# mcp_ml_server.py
from __future__ import annotations

import json
from typing import Any, Dict

from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

from parquet_store import append_audit_row, export_audit_copy
import ml_pipeline as ml


server = Server("mcp-ml-agent")


def _ok(payload: Dict[str, Any]) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(payload, indent=2))]


def _err(message: str) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps({"status": "error", "message": message}, indent=2))]


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="train_models",
            description="Train Logistic Regression and KNN and export joblib artifacts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "test_size": {"type": "number", "default": 0.2},
                    "random_state": {"type": "integer", "default": 42},
                    "knn_neighbors": {"type": "integer", "default": 5},
                    "request_id": {"type": "string"}
                }
            },
        ),
        Tool(
            name="predict_logistic_regression",
            description="Run prediction using the Logistic Regression model artifact.",
            inputSchema={
                "type": "object",
                "properties": {
                    "input_data": {"type": "object", "description": "Feature dictionary"},
                    "request_id": {"type": "string"}
                },
                "required": ["input_data"]
            },
        ),
        Tool(
            name="predict_knn",
            description="Run prediction using the KNN model artifact.",
            inputSchema={
                "type": "object",
                "properties": {
                    "input_data": {"type": "object", "description": "Feature dictionary"},
                    "request_id": {"type": "string"}
                },
                "required": ["input_data"]
            },
        ),
        Tool(
            name="confusion_matrix_metrics",
            description="Compute confusion matrix + accuracy/precision/recall for LR and KNN.",
            inputSchema={
                "type": "object",
                "properties": {
                    "test_size": {"type": "number", "default": 0.2},
                    "random_state": {"type": "integer", "default": 42},
                    "knn_neighbors": {"type": "integer", "default": 5},
                    "request_id": {"type": "string"}
                }
            },
        ),
        Tool(
            name="select_best_model",
            description="Select best model using a rule (default: recall then precision then accuracy).",
            inputSchema={
                "type": "object",
                "properties": {
                    "rule": {"type": "string", "default": "recall_then_precision_then_accuracy"},
                    "test_size": {"type": "number", "default": 0.2},
                    "random_state": {"type": "integer", "default": 42},
                    "knn_neighbors": {"type": "integer", "default": 5},
                    "request_id": {"type": "string"}
                }
            },
        ),
        Tool(
            name="shap_explain_selected",
            description="Run SHAP explanations for the currently selected model (global + local).",
            inputSchema={
                "type": "object",
                "properties": {
                    "local_index": {"type": "integer", "default": 0},
                    "top_k": {"type": "integer", "default": 8},
                    "test_size": {"type": "number", "default": 0.2},
                    "random_state": {"type": "integer", "default": 42},
                    "knn_neighbors": {"type": "integer", "default": 5},
                    "selection_rule": {"type": "string", "default": "recall_then_precision_then_accuracy"},
                    "request_id": {"type": "string"}
                }
            },
        ),
        Tool(
            name="export_parquet_audit",
            description="Export a copy of the Parquet audit log for submission.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "default": "audit_export.parquet"},
                    "request_id": {"type": "string"}
                }
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> list[TextContent]:
    request_id = arguments.get("request_id", "unknown")

    try:
        if name == "train_models":
            result = ml.train_and_export_all(
                test_size=float(arguments.get("test_size", 0.2)),
                random_state=int(arguments.get("random_state", 42)),
                knn_neighbors=int(arguments.get("knn_neighbors", 5)),
            )
            append_audit_row({
                "request_id": request_id,
                "tool": name,
                "status": result["status"],
            })
            return _ok(result)

        if name in ("predict_logistic_regression", "predict_knn"):
            model_name = "logistic_regression" if name == "predict_logistic_regression" else "knn"
            model = ml.load_model(model_name)
            pred = ml.predict_with_model(model, arguments["input_data"])

            response = {
                "status": "success",
                "tool": name,
                "model_used": model_name,
                "result": pred,
            }

            append_audit_row({
                "request_id": request_id,
                "tool": name,
                "model_used": model_name,
                "prediction": pred["prediction"],
                "confidence": pred["confidence"],
                "input_json": json.dumps(arguments["input_data"], sort_keys=True),
            })

            return _ok(response)

        if name == "confusion_matrix_metrics":
            ds = ml.load_dataset(
                test_size=float(arguments.get("test_size", 0.2)),
                random_state=int(arguments.get("random_state", 42)),
            )
            lr = ml.train_logistic_regression(ds)
            knn = ml.train_knn(ds, n_neighbors=int(arguments.get("knn_neighbors", 5)))

            eval_lr = ml.evaluate_model(lr, "logistic_regression", ds)
            eval_knn = ml.evaluate_model(knn, "knn", ds)

            payload = {
                "status": "success",
                "logistic_regression": {
                    "accuracy": eval_lr.accuracy,
                    "precision": eval_lr.precision,
                    "recall": eval_lr.recall,
                    "confusion_matrix": eval_lr.confusion_matrix,
                },
                "knn": {
                    "accuracy": eval_knn.accuracy,
                    "precision": eval_knn.precision,
                    "recall": eval_knn.recall,
                    "confusion_matrix": eval_knn.confusion_matrix,
                },
            }

            append_audit_row({
                "request_id": request_id,
                "tool": name,
                "status": "success",
            })

            return _ok(payload)

        if name == "select_best_model":
            ds = ml.load_dataset(
                test_size=float(arguments.get("test_size", 0.2)),
                random_state=int(arguments.get("random_state", 42)),
            )
            lr = ml.train_logistic_regression(ds)
            knn = ml.train_knn(ds, n_neighbors=int(arguments.get("knn_neighbors", 5)))

            eval_lr = ml.evaluate_model(lr, "logistic_regression", ds)
            eval_knn = ml.evaluate_model(knn, "knn", ds)

            selection = ml.select_model(
                eval_lr,
                eval_knn,
                rule=str(arguments.get("rule", "recall_then_precision_then_accuracy")),
            )

            append_audit_row({
                "request_id": request_id,
                "tool": name,
                "selected_model": selection["selected_model"],
                "rule": selection["rule"],
            })

            return _ok({"status": "success", "selection": selection})

        if name == "shap_explain_selected":
            ds = ml.load_dataset(
                test_size=float(arguments.get("test_size", 0.2)),
                random_state=int(arguments.get("random_state", 42)),
            )
            lr = ml.train_logistic_regression(ds)
            knn = ml.train_knn(ds, n_neighbors=int(arguments.get("knn_neighbors", 5)))

            eval_lr = ml.evaluate_model(lr, "logistic_regression", ds)
            eval_knn = ml.evaluate_model(knn, "knn", ds)

            selection = ml.select_model(
                eval_lr,
                eval_knn,
                rule=str(arguments.get("selection_rule", "recall_then_precision_then_accuracy")),
            )

            selected = selection["selected_model"]
            model = lr if selected == "logistic_regression" else knn

            explanation = ml.shap_explain(
                model=model,
                model_name=selected,
                ds=ds,
                local_index=int(arguments.get("local_index", 0)),
                top_k=int(arguments.get("top_k", 8)),
            )

            append_audit_row({
                "request_id": request_id,
                "tool": name,
                "model_used": selected,
                "status": "success",
            })

            return _ok({"status": "success", "selection": selection, "shap": explanation})

        if name == "export_parquet_audit":
            filename = str(arguments.get("filename", "audit_export.parquet"))
            path = export_audit_copy(filename)

            append_audit_row({
                "request_id": request_id,
                "tool": name,
                "export_path": path,
                "status": "success",
            })

            return _ok({"status": "success", "parquet_path": path})

        return _err(f"Unknown tool: {name}")

    except Exception as e:
        append_audit_row({
            "request_id": request_id,
            "tool": name,
            "status": "error",
            "error": str(e),
        })
        return _err(str(e))


async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())