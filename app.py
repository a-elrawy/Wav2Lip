import subprocess
import os
from fastapi import FastAPI, BackgroundTasks
from librosa.core import audio
from pydantic import BaseModel
import uuid
import requests
from upload import upload_file, make_blob_public
import tempfile
import base64
import time
from inference_func import lip_sync
# import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/lars/ML/remotework-347706-de94e050c66e.json"

bucket_name = "publictestbucket1"

class GenerateRequest(BaseModel):
    audio_url : str = "https://file.notion.so/f/f/c88b8b60-9f5e-4909-b259-0af7f3549da1/2dd5980d-b577-42b3-bab6-785c45a41034/ElevenLabs_2023-09-19T22_54_11_Clyde_pre_s87_sb75_m1.mp3?id=c69120bc-cbb5-4099-bb23-a1b544732583&table=block&spaceId=c88b8b60-9f5e-4909-b259-0af7f3549da1&expirationTimestamp=1695463200000&signature=oYs5lMlhDILQSHOZ3WKsyiCyOSc7auKv9UvmO7eW8dM&downloadName=ElevenLabs_2023-09-19T22_54_11_Clyde_pre_s87_sb75_m1.mp3"
    video_reference_url : str = "https://file.notion.so/f/f/c88b8b60-9f5e-4909-b259-0af7f3549da1/be5a2b75-e1b0-46a3-8925-37bc2148fb55/id10002eNc4LrrvV80005319005444.mp4?id=0731d3e4-f6de-4f52-8884-55ef83185a67&table=block&spaceId=c88b8b60-9f5e-4909-b259-0af7f3549da1&expirationTimestamp=1695643200000&signature=duaNb94mwxUr7r2lMhatSMqwALpL26cffnt1jV_fPfk&downloadName=id10002%23eNc4LrrvV80%23005319%23005444.mp4.mp4"
    text: str
    voiceName: str = "21m00Tcm4TlvDq8ikWAM"
    voice_settings: dict ={
            "stability": 0,
            "similarity_boost": 0
          }
    offset : str = "0"


class TextResponse(BaseModel):
    url : str


# allow CORS
from fastapi.middleware.cors import CORSMiddleware


origins = [
    "http://localhost",
    "http://localhost:8000",
]


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify the domains you want to allow, for development you can use ["*"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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



def process_generate(text, voiceName, voice_settings, video_path, offset, fname):
    # Call the TTS endpoint
    # log time
    og_start = time.time()
    start = time.time()

    tts_response = requests.post('https://ai-interactions-staging.azurewebsites.net/synthesize_speech_eleven_labs', json={
        'text': text,
        'voiceName': voiceName,
        'voice_settings': voice_settings
    })
    tts_response.raise_for_status()
    print(f"Time taken for TTS: {time.time() - start}")
    
    # Save the audio content to a temporary file
    start = time.time()
    audio_content_base64 = tts_response.json().get("audioContent")
    audio_data = base64.b64decode(audio_content_base64)
    audio_path = f"temp_{fname}.mp3"
    with open(audio_path, "wb") as audio_file:
        audio_file.write(audio_data)
    print(f"Time taken for saving audio: {time.time() - start}")
    
    # Now, you can use this audio_path in your existing process
    start = time.time()
    # video_path = download_file(url=video_path, ext=".mp4")
    video_path = "media-.mp4"
    print(f"Time taken for downloading video: {time.time() - start}")
    # log video time
    start = time.time()

    # subprocess.call(f"python inference.py --checkpoint_path {model_path} --face {video_path} --audio {audio_path} --face_det_batch_size 16", shell=True)

    # def lip_sync(face, audio, outfile, fps=30, resize_factor=1, rotate=False, crop=(0, 0, -1, -1),
    #          img_size=96, wav2lip_batch_size=128, face_det_batch_size=16, pads=(0, 20, 0, 0),
    #          static=False, nosmooth=False):

    lip_sync(face=video_path, audio_path=audio_path, outfile=f"results/result_voice.mp4")

    print(f"Time taken for video: {time.time() - start}")
    start = time.time()
    upload_file(bucket_name=bucket_name, source_file_name="./results/result_voice.mp4", destination_blob_name=f"{fname}.mp4")
    make_blob_public(bucket_name=bucket_name, blob_name=f"{fname}.mp4")
    print(f"Time taken for video upload: {time.time() - start}")
    # Cleanup the temporary audio file after processing
    os.remove(audio_path)
    print(f"Time taken for total: {time.time() - og_start}")

@app.post("/generate")
async def generate(generate_request: GenerateRequest, background_tasks: BackgroundTasks):
    fname = str(uuid.uuid4())
    path = f"{fname}.mp4"
    
    background_tasks.add_task(
        process_generate, 
        generate_request.text, 
        generate_request.voiceName, 
        generate_request.voice_settings, 
        generate_request.video_reference_url, 
        generate_request.offset, 
        fname
    )
    
    return TextResponse(url=f"https://storage.googleapis.com/{bucket_name}/{path}")


## https://dl.airtable.com/.attachments/2e1cb4fe82d330a3868b3d2114f102eb/795b6246/german.mp3

## https://dl.airtable.com/.attachments/fbdab15dad324e3fb05d4333a7a62b2a/dfce64fb/static-short.mp4