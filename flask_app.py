

import json
import os
from datetime import datetime

import requests
from flask import Flask, render_template, request, Response, stream_with_context, jsonify

app = Flask(__name__)
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# FastAPI backend URL
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")


def api_get(path, **kwargs):
    """GET request to the FastAPI backend."""
    try:
        resp = requests.get(f"{API_BASE}{path}", timeout=30, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        return {"error": "Backend not reachable. Is the FastAPI server running?"}
    except Exception as e:
        return {"error": str(e)}


def api_post(path, **kwargs):
    """POST request to the FastAPI backend."""
    try:
        resp = requests.post(f"{API_BASE}{path}", timeout=120, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        return {"error": "Backend not reachable. Is the FastAPI server running?"}
    except Exception as e:
        return {"error": str(e)}


# ─── Dashboard ───────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    """Dashboard page with system stats and analytics."""
    # Get health/analytics from backend
    health = api_get("/health/ready")
    analytics = api_get("/analytics")

    # Get recent documents
    docs_resp = api_get("/documents/", params={"size": 5})
    documents = []
    total_docs = 0
    if isinstance(docs_resp, dict):
        documents = docs_resp.get("items", docs_resp.get("documents", []))
        total_docs = docs_resp.get("total", len(documents))
    elif isinstance(docs_resp, list):
        documents = docs_resp
        total_docs = len(documents)

    stats = {
        "health": health,
        "analytics": analytics if "error" not in analytics else {},
        "recent_docs": documents,
        "total_docs": total_docs,
    }
    return render_template("dashboard.html", stats=stats)


# ─── Documents ───────────────────────────────────────────────────────────────

@app.route("/documents")
def documents_page():
    """Document management page with upload and listing."""
    docs_resp = api_get("/documents/", params={"size": 50})
    documents = []
    if isinstance(docs_resp, dict):
        documents = docs_resp.get("items", docs_resp.get("documents", []))
    elif isinstance(docs_resp, list):
        documents = docs_resp
    return render_template(
        "documents.html",
        documents=documents,
        max_upload_mb=MAX_UPLOAD_MB,
    )


@app.route("/documents/upload", methods=["POST"])
def upload_document():
    """Handle document upload via multipart form."""
    if "file" not in request.files:
        return jsonify({"error": "No file selected"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    try:
        resp = requests.post(
            f"{API_BASE}/documents/upload",
            files={"file": (file.filename, file.stream, file.content_type)},
            timeout=60,
        )
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.ConnectionError:
        return jsonify({"error": "Backend not reachable"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/documents/<doc_id>")
def document_detail(doc_id):
    """Document detail page showing chunks, classification, fields."""
    doc = api_get(f"/documents/{doc_id}")
    classification = api_get(f"/documents/{doc_id}/classification")
    fields = api_get(f"/documents/{doc_id}/extracted-fields")
    chunks = api_get(f"/documents/{doc_id}/chunks")

    return render_template(
        "document_detail.html",
        doc=doc,
        classification=classification if "error" not in classification else None,
        fields=fields if "error" not in fields else None,
        chunks=chunks if "error" not in chunks else [],
    )


# ─── Agents ──────────────────────────────────────────────────────────────────

@app.route("/agents")
def agents_page():
    """Agent workflows page."""
    workflows_resp = api_get("/agents/workflows")
    workflows = {}
    if isinstance(workflows_resp, dict) and "workflows" in workflows_resp:
        for wf in workflows_resp["workflows"]:
            workflows[wf["name"]] = {
                "description": wf["description"],
                "required_params": wf["required_params"],
                "optional_params": wf.get("optional_params", []),
            }
    return render_template("agents.html", workflows=workflows)


@app.route("/api/run_agent_stream/<workflow_name>", methods=["POST"])
def run_agent_stream(workflow_name):
    """SSE proxy to stream agent execution from FastAPI backend."""
    data = request.json or {}
    params = data.get("params", {})

    def generate():
        try:
            yield f"data: {json.dumps({'type': 'status', 'content': f'Starting {workflow_name}...'})}\n\n"

            # Call backend agent run endpoint
            resp = requests.post(
                f"{API_BASE}/agents/run",
                json={"workflow": workflow_name, "params": params},
                timeout=300,
            )

            if resp.status_code != 200:
                yield f"data: {json.dumps({'type': 'error', 'content': resp.text})}\n\n"
                return

            result = resp.json()

            # Stream the steps
            import time
            for step in result.get("steps", []):
                yield f"data: {json.dumps({'type': 'step', 'content': step})}\n\n"
                time.sleep(0.1)

            # Final answer
            if result.get("final_answer"):
                yield f"data: {json.dumps({'type': 'result', 'content': result['final_answer']})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'content': 'Workflow completed.'})}\n\n"

        except requests.ConnectionError:
            yield f"data: {json.dumps({'type': 'error', 'content': 'Backend not reachable. Start FastAPI first.'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


# ─── Chat ────────────────────────────────────────────────────────────────────

@app.route("/chat")
def chat_page():
    """RAG Chat page."""
    # Get existing conversations
    conversations = api_get("/conversations")
    if isinstance(conversations, dict) and "error" in conversations:
        conversations = []
    return render_template("chat.html", conversations=conversations)


@app.route("/api/conversations/<conversation_id>/messages")
def conversation_messages_proxy(conversation_id):
    """Proxy conversation history requests to FastAPI backend."""
    result = api_get(f"/conversations/{conversation_id}/messages")
    if isinstance(result, dict) and "error" in result:
        return jsonify(result), 502
    return jsonify(result)


@app.route("/api/chat", methods=["POST"])
def chat_proxy():
    """Proxy chat requests to FastAPI backend."""
    data = request.json or {}
    result = api_post("/chat", json=data)
    return jsonify(result)


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream_proxy():
    """SSE proxy for streaming chat responses."""
    data = request.json or {}

    def generate():
        try:
            resp = requests.post(
                f"{API_BASE}/chat/stream",
                json=data,
                stream=True,
                timeout=120,
            )
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if line:
                    yield f"{line}\n\n"
        except requests.ConnectionError:
            yield f"data: {json.dumps({'type': 'error', 'content': 'Backend not reachable'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


# ─── Search ──────────────────────────────────────────────────────────────────

@app.route("/api/search", methods=["POST"])
def search_proxy():
    """Proxy search requests to FastAPI backend."""
    data = request.json or {}
    result = api_post("/search", json=data)
    return jsonify(result)


# ─── Template filters ────────────────────────────────────────────────────────

@app.template_filter("datetime")
def format_datetime(value):
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%b %d, %Y %H:%M")
        except Exception:
            return value
    return value


@app.template_filter("filesize")
def format_filesize(value):
    try:
        size = int(value)
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    except Exception:
        return value


if __name__ == "__main__":
    print("=" * 60)
    print("  DocIntelli Platform UI")
    print(f"  Backend API: {API_BASE}")
    print(f"  Frontend:    http://localhost:{FLASK_PORT}")
    print("=" * 60)
    print("\nMake sure the FastAPI backend is running:")
    print("  cd backend && uvicorn app.main:app --reload")
    print()
    app.run(debug=FLASK_DEBUG, port=FLASK_PORT)
