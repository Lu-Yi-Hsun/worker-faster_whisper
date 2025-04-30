import base64
import tempfile
import subprocess
import os

from rp_schema import INPUT_VALIDATIONS
from runpod.serverless.utils import download_files_from_urls, rp_cleanup, rp_debugger
from runpod.serverless.utils.rp_validator import validate
import runpod
import predict

MODEL = predict.Predictor()
MODEL.setup()

def base64_to_tempfile(base64_file: str) -> str:
    '''
    Convert base64 file to tempfile.
    '''
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(base64.b64decode(base64_file))
    return temp_file.name

def download_online_audio(job_id: str, url: str) -> str:
    '''
    Use yt-dlp to download audio from YouTube or Vimeo.
    '''
    output_path = f"/tmp/{job_id}.wav"
    try:
        result = subprocess.run([
            "yt-dlp", "-x", "--audio-format", "wav",
            "-o", output_path, url
        ], capture_output=True, text=True, check=True)

        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1024:
            raise Exception("音訊檔下載失敗或大小異常")

        return output_path

    except subprocess.CalledProcessError as e:
        raise Exception(f"yt-dlp 錯誤：{e.stderr.strip()}")
    except Exception as e:
        raise Exception(f"下載/儲存失敗：{str(e)}")

@rp_debugger.FunctionTimer
def run_whisper_job(job):
    '''
    Run inference on the model.
    '''
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
                if any(x in audio_url for x in ["youtube.com", "youtu.be", "vimeo.com"]):
                    audio_input = download_online_audio(job['id'], audio_url)
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
