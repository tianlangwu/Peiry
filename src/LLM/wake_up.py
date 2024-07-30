import pvporcupine

# import sounddevice beucase pyaudio not working on my mac
import sounddevice as sd
import pyaudio
import struct
import os
import sys
from contextlib import contextmanager

# Add the `src` directory to sys.path
# Get the path to the parent directory of the current script's directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Add this directory to sys.path
sys.path.insert(0, parent_dir)
print(f"Parent directory: {parent_dir}")
from config import picovoice_access_key


current_path = os.getcwd()  # Get the current working directory

# keyword_paths = os.path.join(
#     parent_dir,
#     "Models/pvporcupine/嘿-琪琪_zh_mac_v3_0_0/嘿-琪琪_zh_mac_v3_0_0.ppn",
# )

keyword_paths = os.path.join(
    parent_dir,
    "Models/pvporcupine/嘿-琪琪_zh_mac_v3_0_0/嘿-琪琪_zh_mac_v3_0_0.ppn",
)

model_path = os.path.join(
    parent_dir,
    "Models/pvporcupine/嘿-琪琪_zh_mac_v3_0_0/porcupine_params_zh.pv",
)


@contextmanager
def get_audio_stream(sample_rate, channels, format, chunk_size):
    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(
            rate=sample_rate,
            channels=channels,
            format=format,
            input=True,
            frames_per_buffer=chunk_size,
        )
        yield stream
    finally:
        if "stream" in locals():
            stream.stop_stream()
            stream.close()
        pa.terminate()


def detect_wake_word(
    access_key=picovoice_access_key,
    keyword_paths=keyword_paths,
    model_path=model_path,
):
    try:
        porcupine = pvporcupine.create(
            access_key=access_key,
            keyword_paths=[keyword_paths],
            model_path=model_path,
        )
    except Exception as e:
        print(f"Error creating Porcupine instance: {e}")
        return

    print("Listening for wake word...")

    with get_audio_stream(
        porcupine.sample_rate, 1, pyaudio.paInt16, porcupine.frame_length
    ) as stream:
        try:
            while True:
                pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
                pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

                keyword_index = porcupine.process(pcm)
                if keyword_index >= 0:
                    print(f"Wake word detected! (index: {keyword_index})")
                    break
        except KeyboardInterrupt:
            print("Stopping...")
        except Exception as e:
            print(f"Error during wake word detection: {e}")
        finally:
            porcupine.delete()


if __name__ == "__main__":
    detect_wake_word(picovoice_access_key, keyword_paths, model_path)
