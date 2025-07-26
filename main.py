# ---------- KEEP-ALIVE (run first) ----------
import os
import threading
import time
import requests

SELF_URL = os.environ.get("RENDER_EXTERNAL_URL")  # Render injects this
KEEP_ALIVE_PATH = "/"  # or any cheap endpoint (ensure this path exists or use root)
PING_EVERY = 30  # seconds

def _keep_alive():
    if not SELF_URL:  # local dev – skip
        print("SELF_URL not set, skipping keep-alive.") # Optional: Log for debugging
        return
    print(f"Starting keep-alive pings to {SELF_URL}{KEEP_ALIVE_PATH}") # Optional: Log for debugging
    while True:
        try:
            # Ensure there's a single slash between URL and path
            url = f"{SELF_URL.rstrip('/')}{KEEP_ALIVE_PATH}"
            print(f"Pinging {url}") # Optional: Log for debugging
            requests.get(url, timeout=10)
            print("Ping successful") # Optional: Log for debugging
        except Exception as e:
            print(f"Ping failed: {e}") # Optional: Log for debugging
            pass  # ignore hiccups
        time.sleep(PING_EVERY)

# Start the keep-alive thread if SELF_URL is set
if SELF_URL:
    threading.Thread(target=_keep_alive, daemon=True).start()
else:
    print("Render external URL not found. Keep-alive disabled.") # Optional: Log for debugging

# ---------- YOUR ORIGINAL FastAPI APP ----------
# Import necessary modules
from fastapi import FastAPI, HTTPException # Import HTTPException for error handling
import subprocess
import re
import pathlib
# Assuming DPPify is correctly defined in backend.main_agent
from backend.main_agent import DPPify
# Import Pydantic models and types
from pydantic import BaseModel
from typing import Literal # Import Literal from typing (Python 3.8+) or use typing_extensions

# Get the api key from environment variable
api_key = os.environ.get('CEREBRAS_API_KEY')

app = FastAPI()

# Define the input model for the endpoint
class DPPify_input(BaseModel):
    topic_name: str
    question_type: Literal["only MCQ", "only SAQ", "both"]
    total_q: int
    level: Literal["Easy", "Medium", "Hard", "Very hard"]
    dpp_language: Literal["English", "Bengali", "Hindi"]
    additional_instruction: str

# Function to upload PDF and return the download URL
def upload_pdf(file_path: str) -> str:
    try:
        with open(file_path, "rb") as f:
            # --- FIX 1: Corrected the API URL (removed trailing space) ---
            response = requests.post(
                "https://tmpfiles.org/api/v1/upload", # <-- Corrected URL
                files={"file": f},
                timeout=60 # Increased timeout slightly for file upload
            )
        response.raise_for_status() # Raise an exception for bad status codes
        data = response.json()

        # Check if the expected data structure exists
        if "data" not in data or "url" not in data["data"]:
             raise ValueError("Unexpected response format from tmpfiles.org")

        download_url = data["data"]["url"]
        # Modify URL for direct download
        final_url = download_url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
        # Clean up the local file
        os.remove(file_path)
        return final_url
    except requests.exceptions.RequestException as e:
        # Handle potential network/request errors during upload
        print(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload generated PDF")
    except (ValueError, KeyError) as e:
        # Handle unexpected response format or missing keys
        print(f"Error processing upload response: {e}")
        raise HTTPException(status_code=500, detail="Invalid response from file upload service")
    except OSError as e:
        # Handle potential file system errors (e.g., file not found, permission denied)
        print(f"Error accessing or deleting file {file_path}: {e}")
        raise HTTPException(status_code=500, detail="Error processing generated PDF file")


# --- FIX 2: Corrected endpoint name ---
# --- Also made it async def as it's a FastAPI route ---
@app.post("/generate_dpp/") # <-- Corrected endpoint name
async def generate_dpp(agent_inputs: DPPify_input):
    pdf_path = None
    try:
        # Run the DPPify process
        pdf_path = DPPify().run(
            topic_name=agent_inputs.topic_name,
            question_type=agent_inputs.question_type,
            total_q=agent_inputs.total_q,
            level=agent_inputs.level,
            api_key=api_key, # Pass the API key
            dpp_language=agent_inputs.dpp_language,
            additional_instruction=agent_inputs.additional_instruction
        )

        # Check if DPPify.run actually returned a path (defensive)
        if not pdf_path or not os.path.exists(pdf_path):
            raise HTTPException(status_code=500, detail="DPP generation failed or produced no file.")

        # Upload the generated PDF and get the URL
        pdf_url = upload_pdf(pdf_path)
        return {"pdf_url": pdf_url} # Return as JSON, good practice

    except HTTPException: # Re-raise HTTPExceptions we created
        raise
    except Exception as e: # Catch any other unexpected errors during DPP generation or processing
        print(f"Unexpected error in generate_dpp: {e}")
        # Ensure cleanup happens even if upload fails
        if pdf_path and os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
            except OSError:
                pass # Ignore errors during cleanup
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Optional: Add a simple root endpoint for the keep-alive ping if needed
# Or ensure your existing root endpoint (if any) is lightweight
@app.get("/")
async def root():
    return {"message": "Service is running"}
