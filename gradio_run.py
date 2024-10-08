#!/usr/bin/env python3

import sys
import os
import hashlib
from pathlib import Path
import asyncio

from dotenv import load_dotenv

import openai

import assemblyai as aai

import gradio as gr


################################################################################


load_dotenv()


################################################################################


BASE_DIR = Path(__file__).parent.absolute()

PASSWORD_SALT = os.environ["PASSWORD_SALT"]

aai.settings.api_key = os.environ["ASSEMBLYAI_API_KEY"]


################################################################################


async def aai_transcribe(file,
                         diarization,
                         language_code):
    if not file:
        raise gr.Error("Audio file has not been loaded properly")

    config = aai.TranscriptionConfig(speaker_labels=diarization,
                                     language_detection=False,
                                     language_code=language_code,
                                     speech_model=aai.SpeechModel.best)
    transcriber = aai.Transcriber(config=config)
    future = transcriber.transcribe_async(file)
    transcript = await asyncio.wrap_future(future)

    if transcript.status == aai.TranscriptStatus.error:
        return transcript.error

    return transcript.text


async def openai_query(prompt):
    client = openai.AsyncOpenAI()

    try:
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user",
                       "content": prompt}],
            model="o1-preview")

    except openai.BadRequestError as e:
        raise gr.Error(e)

    return chat_completion.choices[0].message.content


def auth(username, password):
    users = {}

    with open(BASE_DIR / "users.txt", "r", encoding="utf-8") as fp:
        for line in fp:
            uname, hash_ = line.split(":", 2)
            users[uname] = hash_.rstrip()

    if username not in users:
        return False

    hash1 = users[username]

    hash2 = hashlib.sha256(
        (password + PASSWORD_SALT).encode("utf-8")).hexdigest()

    return hash1 == hash2


################################################################################


def main(argv=sys.argv):
    with gr.Blocks() as demo:
        with gr.Tab("Silent"):
            with gr.Row():
                with gr.Column(scale=1):
                    in_aai_file = gr.File(
                        label="Audio File",
                        file_types=[".mp3"])

                    in_aai_diarization = gr.Checkbox(
                        label="Diarization (label speakers)",
                        value=True)

                    in_aai_language = gr.Dropdown(
                        label="Language",
                        choices=["ru", "en"],
                        value="ru")

                    btn_aai_submit = gr.Button("Submit")

                with gr.Column(scale=5):
                    out_aai_transcript = gr.TextArea(
                        label="Transcript")

        with gr.Tab("Witness"):
            with gr.Row():
                with gr.Column():
                    in_openai_prompt = gr.TextArea(
                        label="Prompt")

                    btn_openai_submit = gr.Button("Submit")

                with gr.Column():
                    out_openai_response = gr.TextArea(
                        label="Response")

        btn_aai_submit.click(
            fn=aai_transcribe,
            inputs=[in_aai_file,
                    in_aai_diarization,
                    in_aai_language],
            outputs=[out_aai_transcript])

        btn_openai_submit.click(
            fn=openai_query,
            inputs=[in_openai_prompt],
            outputs=[out_openai_response])

    demo.queue(default_concurrency_limit=20)

    demo.launch(root_path="/silentwitness",
                auth=auth)


if __name__ == "__main__":
    sys.exit(main())
