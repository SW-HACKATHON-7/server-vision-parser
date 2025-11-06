# Chat OCR API V2

세션 기반 채팅 스크린샷 OCR 및 대화 분석 API

## 개요

카카오톡 등의 채팅 스크린샷을 업로드하면 OCR을 통해 메시지를 추출하고, 외부 AI API를 통해 대화 내용을 분석하는 시스템입니다.
여러 장의 스크린샷을 업로드하면 자동으로 겹치는 부분을 찾아 병합하고, 전체 대화 흐름을 분석합니다.

## 주요 기능

- 다중 스크린샷 업로드 및 자동 병합
- 채팅 말풍선 감지 및 OCR
- 외부 AI API 연동을 통한 대화 분석
- 스크린샷 기반 메시지 검색
- 다음 대화 예측
- 대화 시뮬레이션 프록시

## API 명세

### 1. 세션 생성

새로운 분석 세션을 생성합니다.

**Endpoint:** `POST /sessions`

**Request Body:** 없음

**Response:**
```json
{
  "session_id": "2ea688cf-315f-4dcb-9580-59262fbe5888",
  "created_at": "2025-01-15T10:30:00.123456",
  "status": "processing"
}
```

### 2. 스크린샷 업로드

생성된 세션에 스크린샷을 업로드합니다. 여러 번 호출하여 여러 장 업로드 가능합니다.

**Endpoint:** `POST /sessions/{session_id}/upload`

**Request:**
- Content-Type: `multipart/form-data`
- Body:
  - `file`: 이미지 파일 (JPG, PNG)

**Response:**
```json
{
  "screenshot_id": "abc123-def456",
  "session_id": "2ea688cf-315f-4dcb-9580-59262fbe5888",
  "upload_order": 1,
  "processed": false,
  "message": "Screenshot uploaded successfully (order: 1)"
}
```

### 3. 세션 처리

업로드된 스크린샷들을 OCR 처리하고 병합한 후, 외부 AI API를 호출하여 분석합니다.

**Endpoint:** `POST /sessions/{session_id}/process`

**Query Parameters:**
- `relationship` (required): 대화 상대와의 관계 (예: "FRIEND", "SUPERIOR", "LOVER")
- `relationship_info` (required): 관계에 대한 추가 정보 (예: "2년 지기", "회사 상사")

**Request:**
```
POST /sessions/2ea688cf-315f-4dcb-9580-59262fbe5888/process?relationship=FRIEND&relationship_info=2년%20지기
```

**Response:**
```json
{
  "session_id": "2ea688cf-315f-4dcb-9580-59262fbe5888",
  "status": "completed",
  "total_screenshots": 2,
  "total_messages": 22,
  "merge_info": {
    "merge_history": [
      {
        "step": 1,
        "overlap_found": true,
        "overlap_length": 5
      }
    ],
    "total_merged": 22
  },
  "external_api_called": true
}
```

### 4. 메시지 조회

세션의 모든 메시지를 조회합니다. AI 분석 결과가 포함됩니다.

**Endpoint:** `GET /sessions/{session_id}/messages`

**Response:**
```json
{
  "session_id": "2ea688cf-315f-4dcb-9580-59262fbe5888",
  "total_messages": 22,
  "total_screenshots": 2,
  "messages": [
    {
      "message_id": "msg-001",
      "text": "오늘 저녁 뭐 먹을까?",
      "speaker": "user",
      "confidence": 0.95,
      "position": {
        "x": 150.0,
        "y": 200.0,
        "width": 180.0,
        "height": 50.0
      },
      "group_id": 1,
      "score": 85.0,
      "emotional_tone": "POSITIVE",
      "impact_score": 2.0,
      "ai_message": "친근하고 적절한 질문입니다.",
      "suggested_alternative": null
    },
    {
      "message_id": "msg-002",
      "text": "아무거나 괜찮아",
      "speaker": "interlocutor",
      "confidence": 0.92,
      "position": {
        "x": 50.0,
        "y": 270.0,
        "width": 160.0,
        "height": 45.0
      },
      "group_id": 2,
      "score": null,
      "emotional_tone": null,
      "impact_score": null,
      "ai_message": null,
      "suggested_alternative": null
    }
  ]
}
```

**Response Fields:**
- `message_id`: 메시지 고유 ID
- `text`: 메시지 내용
- `speaker`: 발화자 ("user" 또는 "interlocutor")
- `confidence`: OCR 신뢰도 (0.0 ~ 1.0)
- `position`: 스크린샷 내 위치 정보
- `group_id`: 연속된 메시지 그룹 ID
- `score`: AI 평가 점수 (0 ~ 100, user 메시지만 해당)
- `emotional_tone`: 감정 톤 (user 메시지만 해당)
- `impact_score`: 영향 점수 (-3 ~ 3, user 메시지만 해당)
- `ai_message`: AI 피드백 (user 메시지만 해당)
- `suggested_alternative`: AI 추천 대안 표현 (user 메시지만 해당)

### 5. 스크린샷으로 메시지 검색

기존 세션에서 특정 스크린샷에 포함된 메시지를 검색합니다.

**Endpoint:** `POST /sessions/{session_id}/search`

**Request:**
- Content-Type: `multipart/form-data`
- Body:
  - `file`: 검색할 이미지 파일

**Response:**
```json
{
  "matched": true,
  "message": "Found 3 matching messages",
  "results": [
    {
      "message_id": "msg-001",
      "text": "오늘 저녁 뭐 먹을까?",
      "speaker": "user",
      "confidence": 0.95,
      "position": {...},
      "group_id": 1
    }
  ]
}
```

### 6. 스크린샷으로 분석 결과 조회

여러 장의 스크린샷을 업로드하여 해당 메시지들의 분석 결과를 조회합니다.
Fuzzy matching을 지원하므로 OCR 결과가 약간 달라도 매칭됩니다.

**Endpoint:** `POST /sessions/{session_id}/view`

**Request:**
- Content-Type: `multipart/form-data`
- Body:
  - `files`: 이미지 파일 배열 (여러 장 업로드 가능)

**Response:**
```json
{
  "session_id": "2ea688cf-315f-4dcb-9580-59262fbe5888",
  "matched": true,
  "total_matched": 15,
  "total_ocr_extracted": 18,
  "messages": [
    {
      "message_id": "msg-001",
      "text": "오늘 저녁 뭐 먹을까?",
      "speaker": "user",
      "confidence": 0.95,
      "position": {...},
      "group_id": 1,
      "score": 85.0,
      "emotional_tone": "POSITIVE",
      "impact_score": 2.0,
      "ai_message": "친근하고 적절한 질문입니다.",
      "suggested_alternative": null
    }
  ]
}
```

### 7. 다음 대화 예측

현재 대화 맥락을 분석하여 다음에 보낼 메시지를 3가지 스타일로 제안합니다.

**Endpoint:** `POST /sessions/{session_id}/predict-next`

**Request Body:** 없음 (세션 정보 자동 사용)

**Response:**
```json
{
  "session_id": "2ea688cf-315f-4dcb-9580-59262fbe5888",
  "relationship": "FRIEND",
  "relationship_info": "2년 지기",
  "total_messages": 22,
  "suggestions": [
    {
      "style": "공감형",
      "text": "아, 그 부분 궁금하셨겠어요. 제가 먼저 말씀드렸어야 했네요.",
      "expected_impact": 2,
      "explanation": "상대의 궁금증을 인정하고 부드럽게 사과하는 응답"
    },
    {
      "style": "해결형",
      "text": "혹시 급하신가요? 지금 바로 확인해볼게요.",
      "expected_impact": 1,
      "explanation": "즉각적인 해결 의지를 보이는 응답"
    },
    {
      "style": "관계형",
      "text": "앗, 제가 미리 공유를 못 드렸네요. 바로 진행 상황 공유드리겠습니다!",
      "expected_impact": 3,
      "explanation": "책임감을 보이며 적극적으로 소통하려는 자세를 보이는 응답"
    }
  ]
}
```

### 8. 대화 시작 (프록시)

외부 AI와 새로운 대화를 시작합니다. AI가 관계에 맞는 첫 메시지를 보냅니다.

**Endpoint:** `POST /start-conversation`

**Request Body:**
```json
{
  "relationship": "연인"
}
```

**Response:**
```json
{
  "message": "요즘 나랑 보내는 시간 어때? 좀 더 함께 하고 싶어.",
  "thread_id": "thread_W2z3FfcGAB3b7vLPJWrwREaF"
}
```

### 9. 대화 이어가기 (프록시)

진행 중인 대화에 사용자 메시지를 보내고 AI의 응답과 평가를 받습니다.

**Endpoint:** `POST /continue-conversation`

**Request Body:**
```json
{
  "message": "요즘 좀 바빠",
  "thread_id": "thread_W2z3FfcGAB3b7vLPJWrwREaF"
}
```

**Response:**
```json
{
  "message": "바쁜 건 이해해. 그래도 네 마음이 궁금한데 조금만 시간 내줄 수 있을까?",
  "response": {
    "emotional_tone": "NEUTRAL",
    "appropriateness_rating": 60,
    "impact_score": 0,
    "review_comment": "상황을 설명하는 표현이지만 상대방의 감정을 고려한 추가 설명이 있으면 좋겠습니다.",
    "suggested_alternative": "요즘 일이 많아서 시간이 부족해. 미안해, 조금만 기다려줄 수 있을까?"
  }
}
```

**Response Fields:**
- `message`: AI의 응답 메시지
- `response.emotional_tone`: 사용자 메시지의 감정 톤
- `response.appropriateness_rating`: 적절성 평가 (0 ~ 100)
- `response.impact_score`: 관계에 미치는 영향 (-3 ~ 3)
- `response.review_comment`: AI 피드백
- `response.suggested_alternative`: 추천 대안 표현

## 사용 예시

### 1. OCR 및 분석 플로우

```bash
# 1. 세션 생성
curl -X POST http://server/sessions

# 2. 스크린샷 업로드 (여러 장)
curl -X POST http://server/sessions/{session_id}/upload \
  -F "file=@screenshot1.jpg"
curl -X POST http://server/sessions/{session_id}/upload \
  -F "file=@screenshot2.jpg"

# 3. 세션 처리 (OCR + 병합 + AI 분석)
curl -X POST "http://server/sessions/{session_id}/process?relationship=FRIEND&relationship_info=2년%20지기"

# 4. 결과 조회
curl -X GET http://server/sessions/{session_id}/messages

# 5. 다음 대화 예측
curl -X POST http://server/sessions/{session_id}/predict-next
```

### 2. 대화 시뮬레이션

```bash
# 1. 대화 시작
curl -X POST http://server/start-conversation \
  -H "Content-Type: application/json" \
  -d '{"relationship": "연인"}'

# 2. 대화 이어가기
curl -X POST http://server/continue-conversation \
  -H "Content-Type: application/json" \
  -d '{"message": "오늘 힘든 하루였어", "thread_id": "thread_xxx"}'
```

## 테스트 클라이언트

### OCR 전체 플로우 테스트
```bash
python test_client.py
```

### 대화 프록시 테스트
```bash
# 간단한 대화 테스트
python test_conversation.py simple

# 여러 관계 유형 테스트
python test_conversation.py multiple

# 긴 대화 테스트
python test_conversation.py long
```

## 설치 및 실행

### 필요 패키지 설치
```bash
pip install -r requirements.txt
```

### 서버 실행
```bash
python main.py
```

서버는 기본적으로 `0.0.0.0:80`에서 실행됩니다.

### 외부 API 설정

`main.py`의 Configuration 섹션에서 외부 AI API URL을 설정합니다:

```python
EXTERNAL_API_URL = "http://localhost:8080/analyze-messages"
SUGGESTION_API_URL = "http://localhost:8080/suggestion-messages"
START_CONVERSATION_URL = "http://localhost:8080/start-conversation"
SEND_MESSAGE_URL = "http://localhost:8080/send-message"
```

외부 API가 없으면 더미 데이터로 동작합니다.

## 데이터베이스

SQLite를 사용하며, `chat_sessions.db` 파일에 저장됩니다.

### 테이블 구조

**sessions**: 세션 정보
- session_id (TEXT, PK)
- created_at (TEXT)
- updated_at (TEXT)
- status (TEXT)
- total_screenshots (INTEGER)
- total_messages (INTEGER)
- relationship (TEXT)
- relationship_info (TEXT)

**screenshots**: 업로드된 스크린샷
- screenshot_id (TEXT, PK)
- session_id (TEXT, FK)
- file_path (TEXT)
- upload_order (INTEGER)
- uploaded_at (TEXT)
- processed (INTEGER)
- image_width (INTEGER)
- image_height (INTEGER)

**messages**: OCR 추출 메시지
- message_id (TEXT, PK)
- session_id (TEXT, FK)
- screenshot_id (TEXT, FK)
- text (TEXT)
- speaker (TEXT)
- confidence (REAL)
- position_x, position_y, position_width, position_height (REAL)
- group_id (INTEGER)
- sequence_order (INTEGER)
- score (REAL)
- emotional_tone (TEXT)
- impact_score (REAL)
- review_comment (TEXT)
- suggested_alternative (TEXT)
- created_at (TEXT)

## 주요 알고리즘

### 스크린샷 병합

여러 장의 스크린샷을 업로드하면 겹치는 메시지를 자동으로 감지하여 병합합니다.
- 최소 2개 이상의 메시지가 겹치면 병합
- 겹치는 부분 이후의 메시지만 추가
- 겹치지 않으면 순서대로 이어붙임

### Fuzzy Matching

`/view` 엔드포인트에서는 SequenceMatcher를 사용한 텍스트 유사도 비교를 수행합니다.
- 유사도 임계값: 85%
- OCR 오차를 고려한 유연한 매칭

## 에러 처리

모든 엔드포인트는 실패 시 적절한 HTTP 상태 코드와 에러 메시지를 반환합니다:

- 400: 잘못된 요청 (세션이 처리되지 않음, 메시지 없음 등)
- 404: 리소스를 찾을 수 없음 (세션 없음)
- 500: 서버 내부 오류
- 502: 외부 API 오류
- 504: 외부 API 타임아웃

```json
{
  "detail": "Session not found"
}
```
