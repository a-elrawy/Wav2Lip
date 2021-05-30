import subprocess
import os
from fastapi import FastAPI, BackgroundTasks
from librosa.core import audio
from pydantic import BaseModel
import uuid
import requests


AIRTABLE_API_KEY = "keyIQFLyLvGiM2waU"

class GenerateRequest(BaseModel):
    audio_url : str
    video_reference_url : str
    offset : str


class TextResponse(BaseModel):
    url : str


app = FastAPI()

model_path = "wav2lip_gan.pth"
# video_path = "output.mp4"
# audio_path = "german.mp3"

def download_file(url : str, ext = ".mp3"):
    fname = "media"
    r = requests.get(url)
    filename = f"{fname}-{ext}"
    with open(filename,"wb") as fp:
        fp.write(r.content)
    return filename



def process_generate(audio_path : str,video_path : str ,offset : str ,fname : str):
    ## Download media
    audio_path = download_file(url = audio_path)
    video_path = download_file(url = video_path)
    # face_path = download_file(url = face_path)

    subprocess.call(f"python inference.py --checkpoint_path {model_path} --face {video_path} --audio {audio_path} --face_det_batch_size 16",shell=True)
    # subprocess.call(f"sudo gsutil -h 'Cache-Control:public, max-age=604800' mv ./output.webm gs://personate-1/{fname}",shell=True)


@app.post("/generate")
async def generate(generate_request : GenerateRequest, background_tasks: BackgroundTasks):
    fname = str(uuid.uuid4())
    background_tasks.add_task(process_generate,generate_request.audio_url,generate_request.video_reference_url,generate_request.offset,fname)
    return TextResponse(url=f"https://storage.googleapis.com/personate-1/{fname}")

## https://dl.airtable.com/.attachments/2e1cb4fe82d330a3868b3d2114f102eb/795b6246/german.mp3

## https://dl.airtable.com/.attachments/fbdab15dad324e3fb05d4333a7a62b2a/dfce64fb/static-short.mp4