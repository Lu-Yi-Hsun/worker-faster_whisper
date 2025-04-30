import base64
import tempfile
import subprocess
import os

from vimeo_downloader import Vimeo
from rp_schema import INPUT_VALIDATIONS
from runpod.serverless.utils import download_files_from_urls, rp_cleanup, rp_debugger
from runpod.serverless.utils.rp_validator import validate
import runpod
import predict

MODEL = predict.Predictor()
MODEL.setup()

def base64_to_tempfile(base64_file: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(base64.b64decode(base64_file))
    return temp_file.name

def download_youtube_audio(job_id: str, url: str) -> str:
    output_path = f"/tmp/{job_id}.%(ext)s"
    try:
        subprocess.run([
            "yt-dlp", "-x", "--audio-format", "wav",
            "-o", output_path, url
        ], capture_output=True, text=True, check=True)

        output_path_final = f"/tmp/{job_id}.wav"
        if not os.path.exists(output_path_final) or os.path.getsize(output_path_final) < 1024:
            raise Exception("音訊檔下載失敗或大小異常")

        return output_path_final

    except subprocess.CalledProcessError as e:
        raise Exception(f"yt-dlp 錯誤：{e.stderr.strip()}")

def download_vimeo_audio(job_id: str, url: str) -> str:
    try:
        v = Vimeo(url)
        stream = v.streams[-1]  # 選最高畫質
        temp_mp4 = f"/tmp/{job_id}.mp4"
        stream.download(download_directory="/tmp", filename=f"{job_id}.mp4")

        # 轉成 wav
        temp_wav = f"/tmp/{job_id}.wav"
        subprocess.run([
            "ffmpeg", "-i", temp_mp4, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", temp_wav
        ], capture_output=True, text=True, check=True)

        if not os.path.exists(temp_wav) or os.path.getsize(temp_wav) < 1024:
            raise Exception("Vimeo 音訊轉換失敗或大小異常")

        return temp_wav

    except Exception as e:
        raise Exception(f"Vimeo 下載/轉檔錯誤：{str(e)}")

@rp_debugger.FunctionTimer
def run_whisper_job(job):
    job_input = job['input']

    with rp_debugger.LineTimer('validation_step'):
        input_validation = validate(job_input, INPUT_VALIDATIONS)
        if 'errors' in input_validation:
            return {"error": input_validation['errors']}
        job_input = input_validation['validated_input']

    if not job_input.get('audio', False) and not job_input.get('audio_base64', False):
        return {'error': 'Must provide either audio or audio_base64'}

    if job_input.get('audio', False) and job_input.get('audio_base64', False):
        return {'error': 'Must provide either audio or audio_base64, not both'}

    try:
        if job_input.get('audio', False):
            with rp_debugger.LineTimer('download_step'):
                audio_url = job_input['audio']
                if "youtube.com" in audio_url or "youtu.be" in audio_url:
                    audio_input = download_youtube_audio(job['id'], audio_url)
                elif "vimeo.com" in audio_url:
                    audio_input = download_vimeo_audio(job['id'], audio_url)
                else:
                    audio_input = download_files_from_urls(job['id'], [audio_url])[0]

        elif job_input.get('audio_base64', False):
            audio_input = base64_to_tempfile(job_input['audio_base64'])

        with rp_debugger.LineTimer('prediction_step'):
            whisper_results = MODEL.predict(
                audio=audio_input,
                model_name=job_input["model"],
                transcription=job_input["transcription"],
                translation=job_input["translation"],
                translate=job_input["translate"],
                language=job_input["language"],
                temperature=job_input["temperature"],
                best_of=job_input["best_of"],
                beam_size=job_input["beam_size"],
                patience=job_input["patience"],
                length_penalty=job_input["length_penalty"],
                suppress_tokens=job_input.get("suppress_tokens", "-1"),
                initial_prompt=job_input["initial_prompt"],
                condition_on_previous_text=job_input["condition_on_previous_text"],
                temperature_increment_on_fallback=job_input["temperature_increment_on_fallback"],
                compression_ratio_threshold=job_input["compression_ratio_threshold"],
                logprob_threshold=job_input["logprob_threshold"],
                no_speech_threshold=job_input["no_speech_threshold"],
                enable_vad=job_input["enable_vad"],
                word_timestamps=job_input["word_timestamps"]
            )

        with rp_debugger.LineTimer('cleanup_step'):
            rp_cleanup.clean(['input_objects'])

        return whisper_results

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": run_whisper_job})

