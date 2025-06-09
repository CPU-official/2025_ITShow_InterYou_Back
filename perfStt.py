import sounddevice as sd
import numpy as np
import whisper

print("모델 로딩 중...")
model = whisper.load_model("medium")
print("모델 로딩 완료")

def record_and_transcribe(duration=10, fs=16000):
    print("\n녹음 시작 ({}초)...".format(duration))
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    print("녹음 완료")
    audio_np = audio.flatten().astype(np.float32) / 32768.0
    print("인식 중...")
    result = model.transcribe(audio_np, language="ko")

    print("결과:")
    print(result["text"].strip())
    print("-" * 50)

try:
    while True:
        record_and_transcribe()
except KeyboardInterrupt:
    print("\n 종료되었습니다.")
