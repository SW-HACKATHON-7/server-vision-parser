# 카카오톡 채팅 말풍선 OCR 분석기

카카오톡 채팅 스크린샷에서 말풍선을 자동으로 감지하고 OCR을 수행하여 JSON 형태로 텍스트를 추출하는 시스템입니다.

## 두 가지 사용 방법

### 1. 간단한 독립 실행형 (추천 - 빠른 테스트용)
서버 없이 `target.png` 또는 원하는 이미지를 바로 처리

### 2. FastAPI 서버 방식
REST API로 여러 클라이언트에서 사용 가능

## 기능

- 채팅 이미지에서 말풍선 자동 감지
- 말풍선 타입 구분 (보낸 메시지/받은 메시지)
- 한글/영어 OCR 지원
- JSON 형태로 결과 반환
- 시각화 이미지 생성

## 설치

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. EasyOCR 모델 다운로드

처음 실행 시 EasyOCR이 자동으로 필요한 모델을 다운로드합니다.

## 사용 방법

---

## 방법 1: 간단한 독립 실행형 (추천)

### 기본 사용 (target.png 처리)

```bash
python simple_chat_ocr.py
```

### 다른 이미지 처리

```bash
python simple_chat_ocr.py chat_screenshot.jpg
```

### 결과 확인

- `output/result.json`: 분석 결과 JSON
- `output/visualization.jpg`: 바운딩 박스가 그려진 시각화 이미지

---

## 방법 2: FastAPI 서버

#### 1. 서버 실행

```bash
python main.py
```

또는

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### 2. API 사용

**cURL 예제:**

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@chat_screenshot.jpg"
```

**Python 예제:**

```python
import requests

url = "http://localhost:8000/analyze"
files = {"file": open("chat_screenshot.jpg", "rb")}

response = requests.post(url, files=files)
result = response.json()

print(f"총 {result['data']['total_messages']}개의 메시지를 찾았습니다.")

for message in result['data']['messages']:
    print(f"[{message['bubble_type']}] {message['text']}")
```

**JavaScript 예제:**

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/analyze', {
  method: 'POST',
  body: formData
})
  .then(response => response.json())
  .then(data => {
    console.log(`총 ${data.data.total_messages}개의 메시지 발견`);
    data.data.messages.forEach(msg => {
      console.log(`[${msg.bubble_type}] ${msg.text}`);
    });
  });
```

#### 3. API 문서 확인

브라우저에서 접속:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 테스트 클라이언트

```bash
python test_client.py chat_screenshot.jpg
```

또는 다른 서버 주소 지정:

```bash
python test_client.py chat_screenshot.jpg http://192.168.1.100:8000
```

## API 응답 형식

### 성공 응답

```json
{
  "status": "success",
  "message": "Successfully analyzed 5 chat messages",
  "data": {
    "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
    "image_path": "uploads/20250105_143022_abc12345.jpg",
    "analyzed_at": "2025-01-05T14:30:25.123456",
    "total_messages": 5,
    "image_size": {
      "width": 1080,
      "height": 1920
    },
    "messages": [
      {
        "id": 1,
        "text": "안녕하세요",
        "confidence": 0.987,
        "bubble_type": "left",
        "position": {
          "x": 50.0,
          "y": 120.0,
          "width": 200.0,
          "height": 60.0
        },
        "sender": "",
        "timestamp": ""
      },
      {
        "id": 2,
        "text": "네 안녕하세요!",
        "confidence": 0.954,
        "bubble_type": "right",
        "position": {
          "x": 600.0,
          "y": 200.0,
          "width": 220.0,
          "height": 65.0
        },
        "sender": "",
        "timestamp": ""
      }
    ]
  }
}
```

### 필드 설명

- `id`: 메시지 순번 (1부터 시작)
- `text`: OCR로 추출한 텍스트
- `confidence`: OCR 신뢰도 (0~1)
- `bubble_type`:
  - `left`: 받은 메시지 (화면 왼쪽)
  - `right`: 보낸 메시지 (화면 오른쪽)
- `position`: 말풍선 위치 (x, y, width, height)

## 프로젝트 구조

```
.
├── simple_chat_ocr.py           # ⭐ 독립 실행형 스크립트 (추천)
├── main.py                      # FastAPI 서버
├── test_client.py               # API 테스트 클라이언트
├── chat_ocr_analyzer.py         # 대체 독립 실행형 스크립트
├── requirements.txt             # 의존성 목록
├── .gitignore                   # Git 무시 파일
├── output/                      # 독립 실행형 결과 저장
│   ├── result.json
│   └── visualization.jpg
├── uploads/                     # FastAPI 업로드 이미지
├── results/                     # FastAPI 분석 결과
│   └── chat_20250105_143022_abc12345/
│       ├── chat_analysis.json
│       ├── visualization.jpg
│       └── original.jpg
└── README.md
```

## 동작 원리

### 1. 말풍선 감지

- 그레이스케일 변환 및 이진화
- 모폴로지 연산으로 노이즈 제거
- 윤곽선 검출
- 크기 및 종횡비 필터링
- 화면 위치로 좌/우 판단 (보낸/받은 메시지)

### 2. OCR 수행

- 각 말풍선 영역(ROI) 추출
- EasyOCR로 텍스트 인식
- 신뢰도 계산

### 3. 결과 반환

- JSON 형태로 구조화
- 시각화 이미지 생성 (바운딩 박스 표시)

## 성능 최적화 팁

1. **GPU 사용**: EasyOCR은 GPU를 지원합니다.
   ```python
   reader = easyocr.Reader(['ko', 'en'], gpu=True)
   ```

2. **배치 처리**: 여러 이미지를 동시에 처리할 경우 비동기 처리 권장

3. **이미지 전처리**: 이미지 품질이 낮을 경우 전처리 추가
   - 샤프닝
   - 노이즈 제거
   - 해상도 조정

## 제한 사항

- 현재는 기본적인 말풍선 감지 알고리즘 사용
- 복잡한 레이아웃이나 겹친 말풍선은 인식률이 낮을 수 있음
- 발신자 이름, 타임스탬프는 별도 로직 필요

## 개선 사항

더 높은 정확도를 위해:

1. **YOLO 모델 사용**: 커스텀 말풍선 감지 모델 학습
2. **텍스트 영역 세분화**: 발신자, 본문, 시간 분리
3. **딥러닝 기반 분류**: 말풍선 타입 자동 분류
4. **후처리**: 맞춤법 검사, 문맥 분석

## 라이선스

MIT License

## 기여

이슈 및 PR을 환영합니다!
