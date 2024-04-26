import subprocess
import random
from flask_cors import CORS
from flask import (
    Flask,
    request,
    send_from_directory,
    jsonify,
    make_response,
)

import pyttsx3
import os
import torch
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

app = Flask(__name__)
ckpt_converter = "OpenVoice-main\\checkpoints\\converter"
device = "cpu"
output_dir = "outputs"
path = "uploads"
directory = os.path.join("wav2lip", "results")
simulationlanguage = {
    "chinese": "OpenVoice-main\\checkpoints\\base_speakers\\ZH",
    "english": "OpenVoice-main\\checkpoints\\base_speakers\\EN",
}
source_se_path = {
    "chinese": "OpenVoice-main/checkpoints/base_speakers/ZH/zh_default_se.pth",
    "english": "OpenVoice-main/checkpoints/base_speakers/EN/en_default_se.pth",
}
languagemp = {
    "chinese": {"女": 0, "男": 7},
    "english": {"女": 1, "男": 6},
    "french": {"女": 3, "男": 4},
    "korean": {"女": 5},
}
from pydub import AudioSegment


def get_duration_pydub(file_path):
    audio_file = AudioSegment.from_file(file_path)
    duration = audio_file.duration_seconds
    return duration


@app.route("/audiodrive", methods=["POST"])
def audiodrive():
    id = str(random.randint(0, 1000))
    video = request.files["video"]
    voice = request.files["voice"]
    print(video.filename, voice.filename)
    _, videoextension = os.path.splitext(video.filename)
    videopath = os.path.join(path, "video" + id + videoextension)
    video.save(os.path.join("wav2lip", videopath))

    _, voiceextension = os.path.splitext(voice.filename)
    voicepath = os.path.join(path, "voice" + id + voiceextension)
    voice.save(os.path.join("wav2lip", voicepath))
    resultpath = "result_" + id + ".mp4"
    finalpath = os.path.join("results", resultpath)
    proc = subprocess.Popen(
        f"cd wav2lip && python inference.py --checkpoint_path wav2lip.pth --face {videopath} --audio {voicepath} --outfile {finalpath}",
        stdin=None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
    )
    outinfo, errinfo = proc.communicate()
    print(outinfo.decode("utf-8"))
    print(errinfo.decode("utf-8"))
    try:
        response = make_response(
            send_from_directory(directory, resultpath, as_attachment=True)
        )
        return response
    except Exception as e:
        return jsonify({"code": "异常", "message": "{}".format(e)})


@app.route("/textdrive", methods=["POST"])
def textdrive():
    id = str(random.randint(0, 1000))
    video = request.files["video"]
    _, videoextension = os.path.splitext(video.filename)
    videopath = os.path.join(path, "video" + id + videoextension)
    video.save(os.path.join("wav2lip", videopath))

    text = request.form.get("text")
    language = request.form.get("language").lower()
    gender = request.form.get("gender")
    print(video.filename, text, language, gender)
    voicepath = os.path.join(path, "voice" + id + ".wav")
    try:
        vid = languagemp[language][gender]
    except Exception as e:
        return jsonify(
            {"code": "异常", "message": f"现阶段无法支持{language}{gender}性发音的生成"}
        )
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.setProperty("volume", 0.6)
    voices = engine.getProperty("voices")

    engine.setProperty("voice", voices[vid].id)
    engine.save_to_file(text, os.path.join("wav2lip", voicepath))
    engine.runAndWait()
    engine.stop()

    resultpath = "result_" + id + ".mp4"
    finalpath = os.path.join("results", resultpath)
    proc = subprocess.Popen(
        f"cd wav2lip && python inference.py --checkpoint_path wav2lip.pth --face {videopath} --audio {voicepath} --outfile {finalpath}",
        stdin=None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
    )
    outinfo, errinfo = proc.communicate()
    print(outinfo.decode("utf-8"))
    print(errinfo.decode("utf-8"))
    try:
        response = make_response(
            send_from_directory(directory, resultpath, as_attachment=True)
        )
        return response
    except Exception as e:
        return jsonify({"code": "异常", "message": "{}".format(e)})


@app.route("/simulationdrive", methods=["POST"])
def simulationdrive():
    id = str(random.randint(0, 1000))
    video = request.files["video"]
    voice = request.files["voice"]
    text = request.form.get("text")
    language = request.form.get("language").lower()
    print(video.filename, voice.filename, text, language)
    _, videoextension = os.path.splitext(video.filename)
    videopath = os.path.join(path, "video" + id + videoextension)
    video.save(os.path.join("wav2lip", videopath))

    _, voiceextension = os.path.splitext(voice.filename)
    voicepath = os.path.join("wav2lip", path, "voice" + id + voiceextension)
    voice.save(voicepath)
    duration = get_duration_pydub(voicepath)
    print(duration)

    try:
        ckpt_base = simulationlanguage[language]
    except:
        return jsonify(
            {"code": "异常", "message": f"现阶段无法支持{language}发音的模拟"}
        )
    tone_color_converter = ToneColorConverter(
        f"{ckpt_converter}/config.json", device=device
    )
    tone_color_converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")
    os.makedirs(output_dir, exist_ok=True)
    reference_speaker = voicepath
    target_se, audio_name = se_extractor.get_se(
        reference_speaker, tone_color_converter, target_dir="processed", vad=True
    )

    base_speaker_tts = BaseSpeakerTTS(f"{ckpt_base}/config.json", device=device)
    base_speaker_tts.load_ckpt(f"{ckpt_base}/checkpoint.pth")
    source_se = torch.load(f"{source_se_path[language]}").to(device)
    save_path = f"{output_dir}/output_{language}_{id}.wav"
    src_path = f"{output_dir}/tmp_{id}.wav"
    base_speaker_tts.tts(
        text, src_path, speaker="default", language=language, speed=0.9
    )
    tone_color_converter.convert(
        audio_src_path=src_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=save_path,
    )
    voicepath = os.path.join(os.getcwd(), save_path)
    resultpath = "result_" + id + ".mp4"
    finalpath = os.path.join("results", resultpath)
    proc = subprocess.Popen(
        f"cd wav2lip && python inference.py --checkpoint_path wav2lip.pth --face {videopath} --audio {voicepath} --outfile {finalpath}",
        stdin=None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
    )
    outinfo, errinfo = proc.communicate()
    print(outinfo.decode("utf-8"))
    print(errinfo.decode("utf-8"))
    try:
        response = make_response(
            send_from_directory(directory, resultpath, as_attachment=True)
        )
        return response
    except Exception as e:
        return jsonify({"code": "异常", "message": "{}".format(e)})


if __name__ == "__main__":
    CORS(app, resources=r"/*")
    CORS(app, supports_credentials=True)
    app.run(port=8000)
