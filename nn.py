import logging
import os
import time
from datetime import datetime
import openai
import streamlit as st
from langchain.callbacks import get_openai_callback  


import logging
import os
# import shutil
from display_cost_and_summary import display_summary
import openai

openai.api_key = os.getenv("OPEN_API_KEY")

logging.basicConfig(filename="log.log", filemode="w", level=logging.INFO)
logger = logging.getLogger(__name__)


def transcribe_audio(audio_file_path: str) -> str:
    logger.info(f"transcribe audio {audio_file_path}")
    with open(audio_file_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe(
            file=audio_file, model="whisper-1", response_format="text", language="en"
        )

    with open("transcription.txt", "w") as f:
        f.write(transcript)

    return transcript


def abstract_summary_extraction(transcription):
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a highly skilled AI trained in language comprehension and summarization. I would like you to read the following text and summarize it into a concise abstract paragraph. Aim to retain the most important points, providing a coherent and readable summary that could  help a person understand the main points of the discussion without needing to read the entire text. Please avoid unnecessary details or tangential points.",
            },
            {"role": "user", "content": transcription},
        ],
    )
    return resp["choices"][0]["message"]["content"]


def key_points_extraction(transcription):
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a proficient AI with a specialty in distilling information into key points. Based on the following text, identify and list the main points that were discussed or brought up. These should be the most important ideas, findings, or topics that are crucial to the essence of the discussion. Your goal is to provide a list that someone could read to quickly understand what was talked about.",
            },
            {"role": "user", "content": transcription},
        ],
    )
    return resp["choices"][0]["message"]["content"]


def action_items_extraction(transcription):
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are an AI expert in analyzing conversations and extracting action items. Please review the text and identify any tasks, assignments, or actions that were agreed upon or mentioned as needing to be done. These could be tasks assigned to specific individuals, or general actions that the group has decided to take. Please list these action items clearly and concisely.",
            },
            {"role": "user", "content": transcription},
        ],
    )
    return resp["choices"][0]["message"]["content"]


def sentiment_analysis(transcription):
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "As an AI with expertise in language ad emotion analysis, your task is to analyze the sentiment of the following text. Please consider the overall tone of the discussion, the emotion conveyed by the language used, and the context in which words and phrases are used. Indicate whether the sentiment is generally positive, negative, or neutral, and provide brief explanations for your analysis where possible.",
            },
            {"role": "user", "content": transcription},
        ],
    )
    return resp["choices"][0]["message"]["content"]


def call_recording_minutes(transcription):
    abstract_summary = abstract_summary_extraction(transcription)
    key_points = key_points_extraction(transcription)
    action_items = action_items_extraction(transcription)
    sentiment = sentiment_analysis(transcription)

    return {
        "abstract_summary": abstract_summary,
        "key_points": key_points,
        "action_items": action_items,
        "sentiment": sentiment,
    }


def save_to_md(call_recording_dict, file_name):
    with open(file_name, "w") as file:
        file.write("# Calling Minutes\n\n")
        file.write("## Abstract Summary\n\n")
        file.write(f"{call_recording_dict['abstract_summary']}\n\n")
        file.write("## Key Points\n\n")
        file.write(f"{call_recording_dict['key_points']}\n\n")
        file.write("## Action Items\n\n")
        file.write(f"{call_recording_dict['action_items']}\n\n")
        file.write("## Sentiment Analysis\n\n")
        file.write(f"{call_recording_dict['sentiment']}\n\n")


def call_recording_main():
    dir_path = "latest_upload"
    for filename in os.listdir(dir_path):
        logger.info(f"Starting Summarizing calls {filename}")
        transcription = transcribe_audio(f"{dir_path}/{filename}")
        minutes = call_recording_minutes(transcription)
        filename = filename.split(".")[0]
        file = f"audio_summary/{filename}_call_recording_summary_of_.md"
        save_to_md(minutes, file)
        display_summary(file)
        return minutes



logging.basicConfig(filename="log.log", filemode="w", level=logging.INFO)
logger = logging.getLogger(__name__)


def file_uploader_placeholder(file_name, file_size):
    st.text(f"{file_name} - {file_size}MB")
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.05)
        my_bar.progress(percent_complete + 1)

def summarizing_audio(file_name):
    st.text(f"Summarizing {file_name}...")
    my_bar = st.progress(0)

    # Initialize the callback for cost and parameter tracking
    with get_openai_callback() as cb:
        summary_data = call_recording_main()
        # Assuming call_recording_main makes all the OpenAI API calls

    for percent_complete in range(100):
        time.sleep(0.05)
        my_bar.progress(percent_complete + 1)

    st.success(f"Summarizing complete for {file_name}!")

   
    st.write("## Cost and API Call Details")
    st.write(f"Total Tokens: {cb.total_tokens}")
    st.write(f"Total Cost (USD): ${cb.total_cost:.4f}")
    st.write(f"Prompt Tokens: {cb.prompt_tokens}")
    st.write(f"Completion Tokens: {cb.completion_tokens}")

    
    for key, value in summary_data.items():
        st.markdown(f"### {key}")
        st.markdown(value)

def uploaded_files_in_dir(uploaded_file):
    today = datetime.today().strftime('%Y-%m-%d')
    file_name = uploaded_file.name
    bytes_data = uploaded_file.getvalue()

    st.write("Filename:", file_name)
    file_size = len(bytes_data) / (1024 * 1024)
    file_uploader_placeholder(file_name, round(file_size, 2))

    with open(os.path.join("latest_upload", f"{today}_{file_name}"), "wb") as f:
        f.write(bytes_data)

    st.audio(bytes_data)
    summarizing_audio(file_name)

def main():
    st.set_page_config(
        page_title="Call Recording Summarization", page_icon="📜"
    )
    st.title("Audio File Uploader")

    uploaded_files = st.file_uploader(
        "Drag and drop or select audio files", type=["mp3", "wav", "ogg", "mp4"], accept_multiple_files=False
    )
    if uploaded_files:
        uploaded_files_in_dir(uploaded_files)
        st.button("Re-run")

if _name_ == "__main__":
    main()