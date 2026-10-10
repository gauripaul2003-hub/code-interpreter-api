import os
import sys
import tempfile
import subprocess
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.environ.get("AIPIPE_TOKEN", ""),
    base_url="https://aipipe.org/openrouter/v1",
)

class CodeRequest(BaseModel):
    code: str

class ErrorAnalysis(BaseModel):
    error_lines: List[int]

def execute_python_code(code: str) -> dict:
    try:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "snippet.py")

            with open(path, "w", encoding="utf-8") as f:
                f.write(code)

            result = subprocess.run(
                [sys.executable, path],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=directory,
            )

        output = result.stdout + result.stderr

        if result.returncode == 0:
            return {"success": True, "output": result.stdout}

        return {"success": False, "output": output}

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": "TimeoutError: execution timed out",
        }

@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):
    execution = execute_python_code(request.code)

    if execution["success"]:
        return {"error": [], "result": execution["output"]}

    prompt = f"""
Analyze the Python code and traceback below.
Return JSON with an integer array named error_lines.
Use 1-based line numbers from the submitted code only.
Do not invent line numbers.

CODE:
{request.code}

TRACEBACK:
{execution["output"]}
"""

    try:
        response = client.chat.completions.create(
            model="google/gemini-2.0-flash-lite-001",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Identify Python error line numbers. "
                        'Return JSON like {"error_lines": [3]}.'
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        import json

        parsed = json.loads(response.choices[0].message.content)
        lines = parsed.get("error_lines", [])
        valid_lines = sorted({
            n for n in lines
            if isinstance(n, int)
            and 1 <= n <= len(request.code.splitlines())
        })

    except Exception:
        valid_lines = []

    return {
        "error": valid_lines,
        "result": execution["output"],
    }
