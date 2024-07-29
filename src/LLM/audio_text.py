import os
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

os.makedirs("../../tmp", exist_ok=True)
project_path = os.getcwd()
model_dir = "iic/SenseVoiceSmall"

model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code="../../sensevoice/model.py",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cpu",
)

# file_path = f"../../tmp/12345.mp3"

# res = model.generate(
#     input=file_path,
#     cache={},
#     language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
#     use_itn=True,
#     batch_size=64,
# )
# text = rich_transcription_postprocess(res[0]["text"])
# print("print out text", text)


# create a def to export the generate
def convert_audio_to_text(file_path):
    targetPath = os.path.join(project_path, file_path)
    print("file_path", file_path)
    res = model.generate(
        input=targetPath,
        cache={},
        language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
        use_itn=True,
        batch_size=64,
    )
    text = rich_transcription_postprocess(res[0]["text"])
    return text
