# MCP Machine Learning Agent

This project demonstrates how to deploy Machine Learning models as **real MCP (Model Context Protocol) tools**. You will train Logistic Regression and KNN models, evaluate them, select the best model, generate SHAP explanations, and log results to Parquet — all exposed through an MCP server.

This project accompanies the MCP and Agent Skills module.

---

# What This Project Includes

The MCP server exposes the following tools:

* Train Logistic Regression and KNN models
* Export models to joblib
* Generate predictions
* Compute confusion matrix and metrics
* Select the best model
* Generate SHAP explanations
* Log tool usage to Parquet
* Export Parquet audit logs

Artifacts generated:

* `artifacts/*.joblib`
* `artifacts/*.json`
* `data_parquet/*.parquet`

---

# Requirements

You must install:

* Python 3.10 or newer
* uv package manager
* VSCode (recommended)

# Setup Instructions

## Step 1 — Clone the repository

In VSCode:

File → Open Folder → select the project folder

OR from terminal:

```bash
git clone <repository_url>
cd mcp_ml_agent
```

---

## Step 2 — Open terminal in the project folder

In VSCode:

Right-click on:

```text
pyproject.toml
```

Select:

```text
Open in Integrated Terminal
```

Confirm you are in the correct folder:

Windows:

```cmd
dir pyproject.toml
```

Mac/Linux:

```bash
ls pyproject.toml
```

---

## Step 3 — Install dependencies

In terminal, run:

If (base) or any conda environment is showing in your command line path:

```bash
conda deactivate
```

```bash
uv sync
```

This will:

* Create `.venv`
* Install all dependencies
* Prepare MCP environment

This may take a few minutes the first time.

---

# Running the MCP Server

Start the MCP server:

```bash
uv run python mcp_ml_server.py
```

If successful, the server will start and wait for MCP tool requests.

No additional output is expected — this is normal.

---

# Running the Client Demo (Optional)

To test the tools automatically:

```bash
uv run python mcp_client_demo.py
```

This will:

* Train models
* Evaluate models
* Select best model
* Generate SHAP explanations
* Export Parquet audit log

---

# Generated Files

After running tools, you will see:

```text
artifacts/
    logistic_regression.joblib
    knn.joblib
    selection.json
    shap_*.json

data_parquet/
    tool_audit.parquet
```

You can open Parquet files in Python:

```python
import pandas as pd

df = pd.read_parquet("data_parquet/tool_audit.parquet")
print(df)
```

---

# Running Jupyter Notebook (Optional)

If using the notebook version:

```bash
uv run jupyter notebook
```

Open the notebook and run the cell.

---

# Important: Do NOT manually activate .venv

Do NOT run:

```bash
.venv\Scripts\activate
```

Always use:

```bash
uv run python filename.py
```

uv automatically uses the correct environment.

---

# Project Structure

```text
mcp_ml_agent/
- pyproject.toml
- mcp_ml_server.py
- mcp_client_demo.py
- ml_pipeline.py
- parquet_store.py

artifacts/
data_parquet/
```
