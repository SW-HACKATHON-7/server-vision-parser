# 외부 API 연동 명세서

## 개요

세션 처리 중 **모든 메시지(user, interlocutor 포함)**에 대해 외부 AI 서버에서 그룹별 분석 결과를 받아옵니다.

### relationship & relationship_info

**relationship**: 대화 상대와의 관계를 나타내는 문자열입니다.
- 예시: `"FRIEND"`, `"SUPERIOR"`, `"COWORKER"`, `"LOVER"` 등
- 클라이언트에서 자유롭게 정의 가능

**relationship_info**: 관계에 대한 추가 정보를 담는 문자열입니다.
- 예시: `"2년 지기"`, `"신입사원"`, `"같은 동아리"`, `"3개월 사귐"` 등
- AI 서버가 대화 맥락을 더 정확히 이해하는 데 사용됩니다.

### group_id란?

**같은 발화자가 연속으로 보낸 메시지 그룹**을 나타냅니다.

예시:
```
[그룹 1] 사용자: "내일 회의 전에 결과 정리해서 공유해 주세요"
[그룹 1] 사용자: "회의가 오전 9시니까 9시 전에 주세요"
[그룹 2] 상대방: "네 알겠습니다"
[그룹 3] 사용자: "다음부터는 눈치있게 일처리 해주세요"
```

- 그룹 1: 사용자가 연속으로 보낸 2개 메시지
- 그룹 2: 상대방 메시지 (interlocutor)
- 그룹 3: 사용자가 다시 보낸 메시지 (새 그룹)

AI 서버는 **그룹 단위**로 분석 결과를 반환합니다.

## API 엔드포인트

```
POST AI_BASE_URL/api/analyze-messages
```

## 요청 형식

### Headers
```http
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY  (선택사항)
```

### Request Body

```json
{
  "relationship": "FRIEND",
  "relationship_info": "2년 지기",
  "messages": [
    {
      "message_id": "uuid-string",
      "text": "사용자가 보낸 메시지 텍스트",
      "speaker": "user",
      "confidence": 0.95,
      "group_id": 1
    },
    {
      "message_id": "another-uuid",
      "text": "또 다른 사용자 메시지",
      "speaker": "user",
      "confidence": 0.87,
      "group_id": 1
    },
    {
      "message_id": "third-uuid",
      "text": "상대방이 보낸 메시지",
      "speaker": "interlocutor",
      "confidence": 0.92,
      "group_id": 2
    },
    {
      "message_id": "fourth-uuid",
      "text": "다시 사용자가 보낸 메시지",
      "speaker": "user",
      "confidence": 0.89,
      "group_id": 3
    }
  ]
}
```

### 필드 설명

| 필드 | 타입 | 설명 |
|------|------|------|
| `relationship` | String | **[필수]** 대화 상대와의 관계 (예: "FRIEND", "SUPERIOR", "COWORKER" 등) |
| `relationship_info` | String | **[필수]** 관계에 대한 추가 정보 (예: "2년 지기", "신입사원" 등) |
| `messages` | Array | **[필수]** 모든 메시지 목록 (user, interlocutor 모두 포함) |
| `messages[].message_id` | String | 메시지 고유 ID (UUID) |
| `messages[].text` | String | 메시지 원문 |
| `messages[].speaker` | String | 발화자 ("user" 또는 "interlocutor") |
| `messages[].confidence` | Float | OCR 신뢰도 (0.0 ~ 1.0) |
| `messages[].group_id` | Integer | 연속 메시지 그룹 ID (같은 그룹 = 연속으로 보낸 메시지) |

## 응답 형식

### Success Response (200 OK)

**HTTP 200 응답이면 성공으로 처리됩니다.** `status` 필드는 선택사항입니다.

**중요**: 응답은 **그룹 단위**로 반환됩니다. 각 그룹의 모든 메시지에 동일한 분석 결과가 적용됩니다.

#### 옵션 1: 배열 직접 반환 (권장)

```json
[
  {
    "group_id": 1,
    "emotional_tone": "NEUTRAL",
    "score": 75,
    "impact_score": 2,
    "review_comment": "상대방을 배려하는 좋은 표현입니다.",
    "suggested_alternative": null
  },
  {
    "group_id": 2,
    "emotional_tone": "NEGATIVE",
    "score": 20,
    "impact_score": -2,
    "review_comment": "다소 공격적으로 들릴 수 있어요.",
    "suggested_alternative": "'그 부분이 조금 의아했는데, 네 생각을 좀 더 듣고 싶어.'라고 말해보세요."
  }
]
```

#### 옵션 2: 객체로 감싸서 반환

```json
{
  "status": "success",
  "response": [
    {
      "group_id": 1,
      "emotional_tone": "NEUTRAL",
      "score": 75,
      "impact_score": 2,
      "review_comment": "상대방을 배려하는 좋은 표현입니다.",
      "suggested_alternative": null
    }
  ]
}
```

### 필드 설명

| 필드 | 타입 | 설명 |
|------|------|------|
| `status` | String | (선택) 응답 상태 ("success" 또는 "error") |
| `response` | Array | (옵션2 사용 시) 각 **그룹**에 대한 분석 결과 배열 |
| `[].group_id` | Integer | **[필수]** 분석 대상 그룹 ID (요청한 messages의 group_id와 매칭) |
| `[].emotional_tone` | String | **[필수]** 감정 톤 ("POSITIVE", "NEUTRAL", "NEGATIVE" 등) |
| `[].score` | Integer | **[필수]** AI가 평가한 점수 (0 ~ 100) |
| `[].impact_score` | Integer | **[필수]** 영향 점수 (-5 ~ +5) |
| `[].review_comment` | String | **[필수]** AI의 피드백 메시지 |
| `[].suggested_alternative` | String \| null | (선택) 대안 표현 제시 (없으면 null) |

### Error Response (4xx, 5xx)

```json
{
  "status": "error",
  "error_code": "RATE_LIMIT_EXCEEDED",
  "message": "API rate limit exceeded. Please try again later."
}
```

## 설정 방법

### 코드에서 외부 API URL 설정

`external_service.py` 파일에서 설정:

```python
from external_service import get_external_service

# 외부 API 사용
service = get_external_service(
    api_url="https://your-ai-server.com/api/analyze-messages",
    api_key="your_api_key_here"
)
```

### 클라이언트에서 세션 처리 호출 예시

세션 처리 시 **relationship과 relationship_info 파라미터는 필수**입니다:

```python
import requests

# 세션 처리 호출
response = requests.post(
    "http://localhost:8000/sessions/{session_id}/process",
    params={
        "relationship": "FRIEND",  # 필수: 관계
        "relationship_info": "2년 지기"  # 필수: 추가 정보
    }
)
```

### main.py 내부에서 사용 예시

```python
# 외부 서버 연동
external_service = get_external_service(
    api_url="https://your-ai-server.com/api/analyze-messages",
    api_key="your_api_key"
)
score_results = await external_service.get_scores_for_messages(
    merged_messages,
    relationship="FRIEND",
    relationship_info="2년 지기"
)
```

## 더미 모드 (개발/테스트용)

외부 API URL을 설정하지 않으면 자동으로 **더미 모드**로 동작합니다:

```python
# api_url=None이면 더미 데이터 생성
service = get_external_service()  # 더미 모드
```

더미 모드에서는:
- 랜덤 score (20 ~ 95)
- 랜덤 emotional_tone (POSITIVE, NEUTRAL, NEGATIVE)
- 랜덤 impact_score (-3 ~ 3)
- 미리 정의된 review_comment 중 랜덤 선택
- 랜덤 suggested_alternative (있거나 null)
- 0.5초 지연 시뮬레이션

## cURL 예시

```bash
curl -X POST https://your-ai-server.com/api/analyze-messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "relationship": "SUPERIOR",
    "relationship_info": "신입사원",
    "messages": [
      {
        "message_id": "123e4567-e89b-12d3-a456-426614174000",
        "text": "내일 회의 전에 결과 정리해서 공유해 주세요",
        "speaker": "user",
        "confidence": 0.95,
        "group_id": 1
      },
      {
        "message_id": "223e4567-e89b-12d3-a456-426614174001",
        "text": "네 알겠습니다",
        "speaker": "interlocutor",
        "confidence": 0.92,
        "group_id": 2
      }
    ]
  }'
```

## Python 요청 예시

```python
import requests

response = requests.post(
    "https://your-ai-server.com/api/analyze-messages",
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer YOUR_API_KEY"
    },
    json={
        "relationship": "FRIEND",  # 필수
        "relationship_info": "2년 지기",  # 필수
        "messages": [
            {
                "message_id": "123e4567-e89b-12d3-a456-426614174000",
                "text": "내일 회의 전에 결과 정리해서 공유해 주세요",
                "speaker": "user",
                "confidence": 0.95,
                "group_id": 1
            },
            {
                "message_id": "223e4567-e89b-12d3-a456-426614174001",
                "text": "회의가 오전 9시니까 9시 전에 주세요",
                "speaker": "user",
                "confidence": 0.92,
                "group_id": 1  # 같은 그룹 (연속 메시지)
            },
            {
                "message_id": "323e4567-e89b-12d3-a456-426614174002",
                "text": "네 알겠습니다",
                "speaker": "interlocutor",
                "confidence": 0.89,
                "group_id": 2  # 상대방 메시지
            }
        ]
    },
    timeout=30
)

if response.status_code == 200:
    data = response.json()
    # 배열이면 그대로 사용, 객체면 'response' 키에서 추출
    results = data if isinstance(data, list) else data.get('response', [])
    for result in results:
        print(f"Group {result['group_id']}: Score={result['score']}, Comment={result['review_comment']}")
```

## 타임아웃 및 에러 처리

- **타임아웃**: 30초
- **실패 시**: 자동으로 더미 데이터로 fallback
- **재시도**: 없음 (1회만 시도)

## 보안 고려사항

1. **HTTPS 사용 필수**
2. **API 키는 환경변수로 관리** (.env 파일 사용 권장)
3. **Rate Limiting 고려** (너무 많은 메시지 한 번에 전송 방지)

## 환경변수 설정 예시

`.env` 파일 생성:

```bash
EXTERNAL_API_URL=https://your-ai-server.com/api/analyze-messages
EXTERNAL_API_KEY=your_secret_api_key_here
```

코드에서 사용:

```python
import os
from dotenv import load_dotenv

load_dotenv()

service = get_external_service(
    api_url=os.getenv("EXTERNAL_API_URL"),
    api_key=os.getenv("EXTERNAL_API_KEY")
)
```

## 데이터베이스 저장

응답받은 데이터는 다음과 같이 데이터베이스에 저장됩니다:

- **group_id**: 해당 그룹의 모든 메시지에 동일한 분석 결과가 적용됨
- **score**: 점수 (0 ~ 100)
- **emotional_tone**: 감정 톤
- **impact_score**: 영향 점수
- **review_comment**: 피드백 메시지
- **suggested_alternative**: 대안 표현 (있는 경우)

같은 group_id를 가진 모든 메시지는 동일한 분석 결과를 공유합니다.
