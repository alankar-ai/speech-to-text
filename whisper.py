import uvicorn
from fastapi import FastAPI, UploadFile, File, Request, Response
import logging
import os
import openai

openai.api_key = os.environ.get('OPENAI_API_KEY')

app = FastAPI()

# Creating log directory
logs_dir = "logs/"
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
logging.basicConfig(filename='logs/whisper.log', level=logging.DEBUG, format='%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

ALLOWED_ORIGINS = '*'


# handle CORS preflight requests
@app.options('/{rest_of_path:path}')
async def preflight_handler(request: Request, rest_of_path: str) -> Response:
    response = Response()
    response.headers['Access-Control-Allow-Origin'] = ALLOWED_ORIGINS
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Authorization, Content-Type'
    return response


# set CORS headers
@app.middleware("http")
async def add_CORS_header(request: Request, call_next):
    response = await call_next(request)
    response.headers['Access-Control-Allow-Origin'] = ALLOWED_ORIGINS
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Authorization, Content-Type'
    return response


@app.get("/")
async def root():
    return {"message": "API is running"}


@app.post("/audio")
async def audio(audio_file: UploadFile = File(...)):
    try:
        logging.info("Processing audio")
        # Save the audio file to a temporary location
        with open("temp_audio.mp3", "wb") as temp_file:
            temp_file.write(await audio_file.read())

        # Transcribe the audio using OpenAI Whisper ASR model
        audio_file = open("temp_audio.mp3", "rb")
        response = openai.Audio.transcribe("whisper-1", audio_file, response_format="text",language="en")
        transcript = response['text']
        
        # Delete the temporary audio file
        os.remove("temp_audio.mp3")

        # Return the transcript in the response
        return {"transcript": transcript}

    except Exception as e:
        logging.exception("An error occurred during audio processing:")
        return {"error": "Failed to process audio"}



if __name__ == '__main__':
    logging.info("*************App Started**************")
    uvicorn.run(app, host='localhost', port=8000)
