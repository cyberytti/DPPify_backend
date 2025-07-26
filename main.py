# ---------- KEEP-ALIVE (run first) ----------
import os
import threading
import time
import requests

SELF_URL = os.environ.get("RENDER_EXTERNAL_URL")     # Render injects this
KEEP_ALIVE_PATH = "/"                           # or any cheap endpoint
PING_EVERY = 30                                 # seconds

def _keep_alive():
    if not SELF_URL:            # local dev – skip
        return
    while True:
        try:
            requests.get(f"{SELF_URL.rstrip('/')}{KEEP_ALIVE_PATH}", timeout=10)
        except Exception:
            pass                # ignore hiccups
        time.sleep(PING_EVERY)

if SELF_URL:
    threading.Thread(target=_keep_alive, daemon=True).start()

# ---------- YOUR ORIGINAL FastAPI APP ----------
from fastapi import FastAPI
import subprocess
import re
import pathlib
from backend.main_agent import DPPify
from pydantic import BaseModel
from typing import Literal

# Get the api key from environment variable
api_key = os.environ.get('CEREBRAS_API_KEY')

app = FastAPI()

class DPPify_input(BaseModel):
    topic_name: str
    question_type: Literal["only MCQ", "only SAQ", "both"]
    total_q: int
    level: Literal["Easy", "Medium", "Hard", "Very hard"]
    dpp_language: Literal["English", "Bengali", "Hindi"]
    additional_instruction: str


def upload_pdf(file_path: str) -> str:
    with open(file_path, "rb") as f:
        response = requests.post(
            "https://tmpfiles.org/api/v1/upload",
            files={"file": f},
            timeout=30
        )
    response.raise_for_status()
    data = response.json()
    download_url = data["data"]["url"]
    os.remove(file_path)
    return download_url.replace("tmpfiles.org", "tmpfiles.org/dl")


@app.post("/genarate_dpp/")
async def genarate_pdf(agent_inputs: DPPify_input):
    pdf_path = DPPify().run(
        topic_name=agent_inputs.topic_name,
        question_type=agent_inputs.question_type,
        total_q=agent_inputs.total_q,
        level=agent_inputs.level,
        api_key=api_key,
        dpp_language=agent_inputs.dpp_language,
        additional_instruction=agent_inputs.additional_instruction
    )

    pdf_url = upload_pdf(pdf_path)
    return pdf_url
