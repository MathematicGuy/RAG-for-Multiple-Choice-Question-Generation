import gradio as gr
import json
from fastapi.testclient import TestClient
from fastapi_app import app as fastapi_app  # import your renamed FastAPI module
import io

# In-process client for the FastAPI app
client = TestClient(fastapi_app)

def read_file_input(file_obj):
    if file_obj is None:
        return None

    # file-like object (has read)
    if hasattr(file_obj, "read"):
        try:
            file_obj.seek(0)
        except Exception:
            pass
        try:
            data = file_obj.read()
            # If read returns str (rare), encode it
            if isinstance(data, str):
                return data.encode()
            return data
        except Exception:
            # continue to other strategies
            pass

    # raw bytes
    if isinstance(file_obj, (bytes, bytearray)):
        return bytes(file_obj)

    # path string (local path)
    if isinstance(file_obj, str):
        try:
            with open(file_obj, "rb") as f:
                return f.read()
        except Exception:
            # not a path, fall through to try encoding string
            return file_obj.encode()

    # dict-like (old Gradio or different frontends)
    try:
        if isinstance(file_obj, dict):
            # common keys: "name", "data"
            if "data" in file_obj:
                data = file_obj["data"]
                if isinstance(data, (bytes, bytearray)):
                    return bytes(data)
                if isinstance(data, str):
                    return data.encode()
            if "name" in file_obj:
                maybe_path = file_obj["name"]
                if isinstance(maybe_path, str):
                    try:
                        with open(maybe_path, "rb") as f:
                            return f.read()
                    except Exception:
                        pass
    except Exception:
        pass

    # Object with attributes (NamedString with .name/.value)
    try:
        name = getattr(file_obj, "name", None)
        data = getattr(file_obj, "data", None)
        value = getattr(file_obj, "value", None)
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        if isinstance(value, str):
            return value.encode()
        if isinstance(name, str):
            try:
                with open(name, "rb") as f:
                    return f.read()
            except Exception:
                pass
    except Exception:
        pass

    # String representation encoded
    try:
        return str(file_obj).encode()
    except Exception:
        return None

def call_generate(file_obj, topics, n_questions, difficulty, question_type):
    if file_obj is None:
        return {"error": "No file uploaded."}

    # Read the uploaded file bytes and create multipart payload
    file_bytes = read_file_input(file_obj)
    if not file_bytes:
        return {"error": "Could not read uploaded file (empty or unknown format)."}
    files = {"file": ("uploaded_file", file_bytes, "application/octet-stream")}

    print(files)

    data = {
        "topics": topics if topics is not None else "",
        "n_questions": str(n_questions),
        "difficulty": difficulty if difficulty is not None else "",
        "question_type": question_type if question_type is not None else ""
    }

    try:
        resp = client.post("/generate/", files=files, data=data, timeout=120)  # increase timeout if needed
    except Exception as e:
        return {"error": f"Request failed: {e}"}
    
    print(resp.status_code)

    if resp.status_code != 200:
        # return helpful debug info
        return {
            "status_code": resp.status_code,
            "text": resp.text,
            "json": None
        }
    
    # print(resp.text)

    # Parse JSON response
    try:
        out = resp.json()
    except Exception:
        # maybe the endpoint returns text: return it directly
        return {"text": resp.text}

    # pretty-format the JSON for display
    return out

# Gradio UI
with gr.Blocks(title="RAG MCQ generator") as gradio_app:
    gr.Markdown("## Upload a file and generate MCQs")

    with gr.Row():
        file_input = gr.File(label="Upload file (PDF, docx, etc)", type="filepath", file_types=[".pdf"])
        topics = gr.Textbox(label="Topics (comma separated)", placeholder="e.g. calculus, derivatives")
    with gr.Row():
        n_questions = gr.Slider(minimum=1, maximum=50, step=1, value=5, label="Number of questions")
        difficulty = gr.Dropdown(choices=["easy", "medium", "hard"], value="medium", label="Difficulty")
        question_type = gr.Dropdown(choices=["mcq", "short", "long"], value="mcq", label="Question type")

    generate_btn = gr.Button("Generate")
    output = gr.JSON(label="Response")

    generate_btn.click(
        fn=call_generate,
        inputs=[file_input, topics, n_questions, difficulty, question_type],
        outputs=[output],
    )

app = gradio_app

if __name__ == "__main__":
    # gradio_app.launch(server_name="0.0.0.0", server_port=7860, share=False)
    gradio_app.launch()
