import os
import json
import tempfile
import asyncio
import whisper
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import google.generativeai as genai
from google.generativeai import types
from dotenv import load_dotenv
from fastapi import HTTPException, FastAPI, UploadFile, File
from pydantic import BaseModel, Field
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# .env 파일에서 환경 변수 로드
load_dotenv()

app = FastAPI(
    title="InterYou Backend API",
    description="AI 분석 + 음성 인식 + 사용자 관리 백엔드"
)

# --- CORS 설정 ---
origins = [
    "https://interyou.mirim-it-show.site",
    "http://15.164.99.18:3000",
    "http://15.164.99.18:3001",
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:5500",
    "null"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- CORS 끝 ---

# --- Gemini 연결 ---
gemini_api_key = os.getenv("GOOGLE_API_KEY")
if not gemini_api_key:
    raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
genai.configure(api_key=gemini_api_key)

model = genai.GenerativeModel(
    'models/gemini-2.5-flash',
    safety_settings=[
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
)
# --- Gemini 끝 ---

# --- 영상 파일 저장
# 업로드 파일 저장 디렉토리

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 영상(또는 오디오/이미지) 파일 업로드 엔드포인트
@app.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    filename = video.filename
    save_path = os.path.join(UPLOAD_DIR, filename)
    # 대용량 수신 처리
    with open(save_path, "wb") as f:
        f.write(await video.read())
    # 클라이언트에 저장 경로 반환
    return JSONResponse(content={"videoUrl": f"/uploads/{filename}"})

# 정적 파일 서비스 (업로드 경로)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


# --- MongoDB 연결 ---
MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "my_database")

client = None
db = None
collection = None

@app.on_event("startup")
async def startup_event():
    global client, db, collection, whisper_model, model_loaded
    # MongoDB 연결
    try:
        client = MongoClient(MONGO_HOST, MONGO_PORT)
        db = client[MONGO_DB_NAME]
        collection = db["users"]
        client.admin.command('ping')
        print("MongoDB 연결 성공!")
    except ConnectionFailure as e:
        print(f"MongoDB 연결 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Database connection error: {e}")

    # Whisper 모델 로드
    try:
        whisper_model = whisper.load_model("small")
        model_loaded = True
        print("Whisper 모델 로드 완료.")
    except Exception as e:
        print("Whisper 모델 로드 실패:", e)
        model_loaded = False

@app.on_event("shutdown")
async def shutdown_event():
    if client:
        client.close()
        print("MongoDB 연결 종료.")

# --- Pydantic 모델 ---
class User(BaseModel):
    name: str
    score: int = 0

class AspectScore(BaseModel):
    aspect: str
    score: float

class AnalyzeRequest(BaseModel):
    question1: str
    answer_text1: str
    question2: str
    answer_text2: str
    question3: str
    answer_text3: str

class AnalyzeResponse(BaseModel):
    answer1_comment: str
    answer1_detailed_score: list[AspectScore]
    answer2_comment: str
    answer2_detailed_score: list[AspectScore]
    answer3_comment: str
    answer3_detailed_score: list[AspectScore]
    total_score: float

class TranscribeResponse(BaseModel):
    transcribed_text: str
# --- 모델 끝 ---

# --- 사용자 관리 API ---
@app.post("/api/users/")
async def create_user(user: User):
    existing_user = collection.find_one({"name": user.name})
    if existing_user:
        raise HTTPException(status_code=409, detail="User with this name already exists")
    user_data = user.dict()
    result = collection.insert_one(user_data)
    return {"message": "User saved successfully", "id": str(result.inserted_id)}

@app.get("/api/ranking/")
async def get_all_users():
    users = list(collection.find({}, {"_id": 0, "name": 1, "score": 1}).sort("score", -1))
    return {"users": users}

# --- Gemini 분석 API ---
@app.post("/api/analyze_answers", response_model=AnalyzeResponse)
async def analyze_answers(request: AnalyzeRequest):
    prompt = f"""당신은 AI 면접 심사위원입니다.
당신은 참가자의 답변 3개를 **창의성, 논리성** 기준으로 평가합니다.

---

### 평가 기준

1. **창의성 (Creativity):** 답변이 얼마나 독창적이고 새로운 관점을 제시했는가
2. **논리성 (Logic):** 답변이 얼마나 합리적이고 일관성 있게 전개되었는가

---

### 유의사항
참가자의 답변들은 **음성 인식 결과**이므로 일부 단어나 문장 구조가 부정확할 수 있습니다.  
발음 오류, 단어 생략, 이상한 문장 등은 감안하되,  
참가자의 **의도와 맥락**을 최대한 고려해 평가해 주세요.

---

### 참가자 답변

Q1: {request.question1}  
A1: {request.answer_text1}

Q2: {request.question2}  
A2: {request.answer_text2}

Q3: {request.question3}  
A3: {request.answer_text3}

---

### 평가 방식

- 각 답변에 대해 **창의성, 논리성** 점수를 부여합니다.  
- 점수는 반드시 **30, 40, 50, 60, 70, 80, 90, 100** 중 하나여야 합니다.  
- 답변의 수준 차이를 반영하여 **답변마다 점수가 다르게** 나오도록 평가하세요.  
- 전체 6개 점수의 평균을 총점으로 계산하고, **소수점 둘째 자리까지** 표현하세요.  
- **모든 답변의 점수가 동일하거나 거의 비슷하지 않도록 유의하세요.**

---

### 결과 출력 형식 (이 형식 외의 문장은 절대 포함하지 마세요)

```json
{{
  "answer1_comment": "<첫 번째 답변에 대한 구체적 피드백>",
  "answer1_detailed_score": [
    {{ "aspect": "창의성", "score": <정수> }},
    {{ "aspect": "논리성", "score": <정수> }}
  ],
  "answer2_comment": "<두 번째 답변에 대한 구체적 피드백>",
  "answer2_detailed_score": [
    {{ "aspect": "창의성", "score": <정수> }},
    {{ "aspect": "논리성", "score": <정수> }}
  ],
  "answer3_comment": "<세 번째 답변에 대한 구체적 피드백>",
  "answer3_detailed_score": [
    {{ "aspect": "창의성", "score": <정수> }},
    {{ "aspect": "논리성", "score": <정수> }}
  ],
  "total_score": <실제 평균값(예: 76.67)>
}}
"""

    try:
        # Gemini 호출
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"},
        )

        # 응답 본문을 JSON으로 파싱
        parsed = json.loads(response.text)

        # Pydantic 모델에 맞게 변환하여 반환
        return AnalyzeResponse(**parsed)

    except json.JSONDecodeError as e:
        # 모델이 JSON 외의 문자를 출력한 경우
        raise HTTPException(status_code=500, detail=f"AI 응답 파싱 오류: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI 분석 중 오류: {e}")


# --- Whisper 음성 인식 API ---
@app.post("/api/transcribe_audio_file", response_model=TranscribeResponse)
async def transcribe_audio_file(audio_file: UploadFile = File(...)):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Whisper 모델이 아직 로드되지 않았습니다.")
    if not audio_file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="오디오 파일만 업로드 가능합니다.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        file_data = await audio_file.read()
        temp_file.write(file_data)
        temp_path = temp_file.name

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: whisper_model.transcribe(temp_path, language="ko"))
        text = result["text"].strip()
        return TranscribeResponse(transcribed_text=text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"음성 인식 실패: {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# --- React 빌드 폴더 연결 ---
build_path = os.path.join(os.path.dirname(__file__), "../2025_ITShow_InterYou_Front/build")
app.mount("/", StaticFiles(directory=build_path, html=True), name="static")

@app.get("/{full_path:path}")
async def serve_react_app(full_path: str):
    return FileResponse(os.path.join(build_path, "index.html"))

# --- 사용자 점수 업데이트 API (작동 안함) --
# --- 사용자 점수 업데이트용 Pydantic 모델 ---
class UpdateScoreRequest(BaseModel):
    name: str
    score: float

# --- 사용자 점수 업데이트 API ---
@app.put("/api/users/score/")
async def update_user_score(data: UpdateScoreRequest):
    name = data.name
    score = data.score

    user = collection.find_one({"name": name})
    if not user:
        raise HTTPException(status_code=404, detail=f"'{name}' 사용자를 찾을 수 없습니다.")

    try:
        result = collection.update_one({"name": name}, {"$set": {"score": score}})
        if result.modified_count == 0:
            return {"message": "점수가 변경되지 않았습니다."}
        return {"message": "점수 업데이트 성공", "name": name, "new_score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"점수 업데이트 실패: {e}")


# --- 서버 실행 ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=3001, reload=True)