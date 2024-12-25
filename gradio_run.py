#!/usr/bin/env python3

import sys
import os
import hashlib
from pathlib import Path
import asyncio

from dotenv import load_dotenv

import openai
import anthropic

import assemblyai as aai

import gradio as gr


################################################################################


load_dotenv()


################################################################################


BASE_DIR = Path(__file__).parent.absolute()

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

PASSWORD_SALT = os.environ["PASSWORD_SALT"]

VICTIM_CONCURRENCY = 5

aai.settings.api_key = os.environ["ASSEMBLYAI_API_KEY"]
aai.settings.http_timeout = 120  # seconds

LLM_GPT_4O = "gpt-4o-2024-11-20"
LLM_CLAUDE = "claude-3-5-sonnet-20241022"

GRADIO_CONCURRENCY_LIMIT = 5

GRADIO_ROOT_PATH = os.environ.get("GRADIO_ROOT_PATH", "/silentwitness")


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
        raise gr.Error(transcript.error)

    parts = []

    if transcript.utterances:
        for utterance in transcript.utterances:
            parts.append(f"Speaker {utterance.speaker}:\n{utterance.text}")

    return "\n\n".join(parts)


async def llm_query(prompt, model):
    if model.startswith("gpt") or model.startswith("o1"):
        client = openai.AsyncOpenAI()

        try:
            response = await client.chat.completions.create(
                messages=[{"role": "user",
                           "content": prompt}],
                model=model)

        except openai.BadRequestError as e:
            raise gr.Error(e)

        try:
            raise gr.Error(response.error)
        except AttributeError:
            pass

        return response.choices[0].message.content

    if model.startswith("claude"):
        client = anthropic.AsyncAnthropic(
            api_key=ANTHROPIC_API_KEY,
            timeout=120.0)

        try:
            response = await client.messages.create(
                model=model,
                messages=[{"role": "user",
                           "content": prompt}],
                temperature=0,
                max_tokens=4096)

        except anthropic.AnthropicError as e:
            raise gr.Error(e)

        try:
            raise gr.Error(response.error.message)
        except AttributeError:
            pass

        return response.content[0].text

    alueError(model)


async def gather_with_concurrency(n, *coros):
    semaphore = asyncio.Semaphore(n)

    async def semaphore_coro(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(semaphore_coro(coro) for coro in coros))


async def victim_query(files, sep, model, prompt_template):
    prompts = []

    for file in files:
        file_content = open(file, encoding="utf-8").read().strip()

        if sep:
            chunks = file_content.split(sep)
        else:
            chunks = [file_content]

        for chunk in chunks:
            prompts.append(prompt_template.format(content=chunk))

    if model.startswith("gpt") or model.startswith("o1"):
        client = openai.AsyncOpenAI()

        chats = []

        kwargs = {}

        if model.startswith("gpt"):
            kwargs["temperature"] = 0.0

        for prompt in prompts:
            chats.append(client.chat.completions.create(
                messages=[{"role": "user",
                           "content": prompt}],
                model=model,
                **kwargs))

        try:
            chat_completions = await gather_with_concurrency(VICTIM_CONCURRENCY, *chats)
            responses = [x.choices[0].message.content for x in chat_completions]
        except Exception as e:
            raise gr.Error(e)

        return "\n\n".join(responses)

    if model.startswith("claude"):
        client = anthropic.AsyncAnthropic(
            api_key=ANTHROPIC_API_KEY,
            timeout=120.0)

        chats = []

        for prompt in prompts:
            chats.append(client.messages.create(
                messages=[{"role": "user",
                           "content": prompt}],
                model=model,
                temperature=0,
                max_tokens=4096))

        try:
            chat_completions = await gather_with_concurrency(VICTIM_CONCURRENCY, *chats)
            responses = [x.content[0].text for x in chat_completions]
        except Exception as e:
            raise gr.Error(e)

        return "\n\n".join(responses)

    raise ValueError(model)

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
                    in_llm_prompt = gr.TextArea(
                        label="Prompt",
                        value="""\
Помоги управленческому консультанту составить резюме звонка с клиентом (Агроэко). \
Особый акцент сделай на стратегических вызовах, целях, задачах, проектах, \
болевых точках и КПЭ (KPI). Встречу консультанты (Александр Каленик) вели \
с директором по маркетингу Агроэко (Дехаев Борис). Суммируй ТОЛЬКО то, \
что говорил КЛИЕНТ, не учитывай гипотезы и варианты, выдвинутые консультантами. \
Тебе будет предоставлен транскрипт записи встречи. Учти, что транскрипт может \
содержать ошибки распознавания речи. Резюме должно быть на русском языке.

<транскрипт>
...
</транскрипт>
""")
                    in_llm_model = gr.Dropdown(
                        label="Model",
                        choices=["o1-preview",
                                 LLM_GPT_4O,
                                 LLM_CLAUDE],
                        value="o1-preview")

                    btn_llm_submit = gr.Button("Submit")

                with gr.Column():
                    out_llm_response = gr.TextArea(
                        label="Response")

        with gr.Tab("Victim"):
            with gr.Row():
                with gr.Column():
                    in_vic_files = gr.Files(
                        label="Text files",
                        file_types=[".txt"])

                    in_vic_sep = gr.Textbox(
                        label="Separator",
                        value="")

                    in_vic_prompt = gr.TextArea(
                        label="Prompt",
                        value="{content}")

                    in_vic_model = gr.Dropdown(
                        label="Model",
                        choices=[LLM_GPT_4O,
                                 "o1-preview",
                                 LLM_CLAUDE],
                        value=LLM_GPT_4O)

                    btn_vic_submit = gr.Button("Submit")

                with gr.Column():
                    out_vic_response = gr.TextArea(
                        label="Result")


        btn_aai_submit.click(
            fn=aai_transcribe,
            inputs=[in_aai_file,
                    in_aai_diarization,
                    in_aai_language],
            outputs=[out_aai_transcript])

        btn_llm_submit.click(
            fn=llm_query,
            inputs=[in_llm_prompt,
                    in_llm_model],
            outputs=[out_llm_response])

        btn_vic_submit.click(
            fn=victim_query,
            inputs=[in_vic_files,
                    in_vic_sep,
                    in_vic_model,
                    in_vic_prompt],
            outputs=[out_vic_response])

    demo.queue(default_concurrency_limit=GRADIO_CONCURRENCY_LIMIT)

    demo.launch(root_path=GRADIO_ROOT_PATH,
                auth=auth)


if __name__ == "__main__":
    sys.exit(main())
