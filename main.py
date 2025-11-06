"""
Chat OCR API V2 - Session-based Multi-screenshot Management
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Path as FastAPIPath, Query, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import cv2
import numpy as np
import easyocr
import uuid
import os
from datetime import datetime
from pathlib import Path
import aiofiles
import shutil

# 로컬 모듈
from database import Database
from merge_logic import (
    merge_multiple_screenshots,
    deduplicate_messages,
    assign_global_group_ids
)
from external_service import get_external_service

# OCR 함수들 (main_old.py에서 통합)
# get_ocr_reader, detect_chat_bubbles, extract_text_from_roi 등은 아래에 정의됨


# ========== Pydantic Models ==========

class SessionCreateResponse(BaseModel):
    """세션 생성 응답"""
    session_id: str
    created_at: str
    status: str


class ScreenshotUploadResponse(BaseModel):
    """스크린샷 업로드 응답"""
    screenshot_id: str
    session_id: str
    upload_order: int
    processed: bool
    message: str


class MessageModel(BaseModel):
    """메시지 모델"""
    message_id: str
    text: str
    speaker: str
    confidence: float
    position: Dict[str, float]
    group_id: Optional[int] = None
    score: Optional[float] = None
    emotional_tone: Optional[str] = None
    impact_score: Optional[float] = None
    ai_message: Optional[str] = None


class SessionMessagesResponse(BaseModel):
    """세션 메시지 조회 응답"""
    session_id: str
    total_messages: int
    total_screenshots: int
    messages: List[MessageModel]


class ProcessSessionResponse(BaseModel):
    """세션 처리 응답"""
    session_id: str
    status: str
    total_screenshots: int
    total_messages: int
    merge_info: Dict[str, Any]
    external_api_called: bool


# ========== Configuration ==========

# 외부 API 설정
EXTERNAL_API_URL = "https://db1ef587c833.ngrok-free.app/analyze-messages"
EXTERNAL_API_KEY = None

print(f"🔧 External API 설정:")
if EXTERNAL_API_URL:
    print(f"  - URL: {EXTERNAL_API_URL}")
    print(f"  - API Key: {'설정됨' if EXTERNAL_API_KEY else '없음'}")
else:
    print(f"  - 더미 모드 (EXTERNAL_API_URL 없음)")


# ========== FastAPI App ==========

app = FastAPI(
    title="Chat OCR API V2",
    description="세션 기반 다중 스크린샷 병합 및 OCR 분석 API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
db = Database()
ocr_reader = None
UPLOAD_DIR = Path("uploads_v2")
UPLOAD_DIR.mkdir(exist_ok=True)


# ========== OCR Functions (최적화됨) ==========

def get_ocr_reader():
    """OCR 리더 싱글톤 (최적화 버전)"""
    global ocr_reader
    if ocr_reader is None:
        print("Initializing EasyOCR Reader (Optimized)...")
        ocr_reader = easyocr.Reader(
            ['ko', 'en'],
            gpu=False,  # CPU 사용 (macOS에서 더 빠름)
            download_enabled=False,  # 모델 재다운로드 방지
            verbose=False  # 로그 줄이기
        )
        print("EasyOCR Reader initialized successfully")
    return ocr_reader


def is_ui_element_or_noise(text: str, bubble: Dict[str, Any]) -> bool:
    """UI 요소나 노이즈 텍스트 필터링"""
    import re
    text_clean = text.strip()

    # UI 요소
    ui_keywords = ['TALK', '메시지 입력', 'LTE', '검색', '설정']
    if any(keyword in text_clean for keyword in ui_keywords):
        return True

    # 순수 시간만 있는 경우
    if len(text_clean) < 15:
        time_match = re.search(r'(오전|오후|AM|PM)?\s*\d{1,2}[:\.]?\d{2}', text_clean)
        if time_match:
            remaining = text_clean.replace(time_match.group(), '').strip()
            if len(remaining) <= 3:
                return True

    # 너무 짧은 텍스트
    if len(text_clean) <= 2:
        return True

    # 날짜 패턴
    if re.match(r'\d{4}년\s+\d{1,2}월\s+\d{1,2}일', text_clean):
        return True

    return False


def is_repeated_sender_name(text: str, bubble: Dict[str, Any], previous_messages: List[Dict]) -> bool:
    """반복되는 발신자 이름 필터링"""
    if bubble['bubble_type'] == 'left' and bubble['width'] < 80 and len(text) <= 4:
        recent_texts = [msg['text'] for msg in previous_messages[-3:]]
        if recent_texts.count(text) >= 2:
            return True
        if bubble['height'] < 30 and len(text) <= 5:
            return True
    return False


def detect_chat_bubbles(image: np.ndarray) -> tuple:
    """이미지에서 채팅 말풍선 영역 감지"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    is_dark_mode = avg_brightness < 100

    if is_dark_mode:
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"  - 총 {len(contours)}개 윤곽선 발견")

    bubbles = []
    img_height, img_width = image.shape[:2]
    filtered_count = {'size': 0, 'area': 0, 'aspect': 0, 'ignored_region': 0}

    TOP_IGNORE_PX = 200
    BOTTOM_IGNORE_PX = 200
    effective_top = TOP_IGNORE_PX
    effective_bottom = img_height - BOTTOM_IGNORE_PX

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if y < effective_top or (y + h) > effective_bottom:
            filtered_count['ignored_region'] += 1
            continue

        if w < 40 or h < 15:
            filtered_count['size'] += 1
            continue
        if w > img_width * 0.95 or h > img_height * 0.4:
            filtered_count['size'] += 1
            continue

        area = cv2.contourArea(contour)
        if area < 500:
            filtered_count['area'] += 1
            continue

        aspect_ratio = w / h
        if aspect_ratio < 0.3 or aspect_ratio > 15:
            filtered_count['aspect'] += 1
            continue

        center_x = x + w / 2
        bubble_type = 'right' if center_x > img_width * 0.5 else 'left'

        bubbles.append({
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'bubble_type': bubble_type
        })

    bubbles.sort(key=lambda b: b['y'])
    print(f"  - 필터링됨: 크기({filtered_count['size']}), 면적({filtered_count['area']}), 종횡비({filtered_count['aspect']}), 무시된 영역({filtered_count['ignored_region']})")
    print(f"  - 병합 전 말풍선: {len(bubbles)}개")

    merged_bubbles = merge_nearby_bubbles(bubbles, img_width, img_height)
    print(f"  - 병합 후 말풍선: {len(merged_bubbles)}개")

    return merged_bubbles, binary


def merge_nearby_bubbles(bubbles: List[Dict[str, Any]], img_width: int, img_height: int) -> List[Dict[str, Any]]:
    """인접한 말풍선 병합"""
    if not bubbles:
        return []

    filtered = []
    for b in bubbles:
        aspect = b['width'] / b['height'] if b['height'] > 0 else 0
        if 0.8 < aspect < 1.2 and b['width'] > 60 and b['height'] > 60:
            continue
        filtered.append(b)

    bubbles = sorted(filtered, key=lambda b: (b['y'], b['x']))

    groups = []
    used = [False] * len(bubbles)

    for i in range(len(bubbles)):
        if used[i]:
            continue

        current_group = [bubbles[i]]
        used[i] = True
        queue = [bubbles[i]]
        head = 0

        while head < len(queue):
            current_bubble = queue[head]
            head += 1

            for j in range(len(bubbles)):
                if used[j]:
                    continue

                other_bubble = bubbles[j]
                y_center_current = current_bubble['y'] + current_bubble['height'] / 2
                y_center_other = other_bubble['y'] + other_bubble['height'] / 2

                is_vertically_close = abs(y_center_current - y_center_other) < (current_bubble['height'] + other_bubble['height']) / 2
                x_dist = max(0, max(current_bubble['x'], other_bubble['x']) - min(current_bubble['x'] + current_bubble['width'], other_bubble['x'] + other_bubble['width']))
                is_horizontally_close = x_dist < 100

                if is_vertically_close and is_horizontally_close:
                    current_group.append(other_bubble)
                    used[j] = True
                    queue.append(other_bubble)

        groups.append(current_group)

    merged = []
    for group in groups:
        if not group:
            continue

        min_x = min(b['x'] for b in group)
        max_x = max(b['x'] + b['width'] for b in group)
        min_y = min(b['y'] for b in group)
        max_y = max(b['y'] + b['height'] for b in group)

        center_x = (min_x + max_x) / 2
        bubble_type = 'right' if center_x > img_width * 0.5 else 'left'

        merged.append({
            'x': min_x,
            'y': min_y,
            'width': max_x - min_x,
            'height': max_y - min_y,
            'bubble_type': bubble_type
        })

    return merged


def extract_text_from_roi(image: np.ndarray, bubble: Dict[str, Any], reader) -> tuple:
    """ROI 영역에서 텍스트 추출 (최적화됨)"""
    x, y, w, h = bubble['x'], bubble['y'], bubble['width'], bubble['height']

    padding = 5
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + w + padding)
    y2 = min(image.shape[0], y + h + padding)

    roi = image[y1:y2, x1:x2]

    if roi.size == 0:
        return "", 0.0

    try:
        results = reader.readtext(
            roi,
            paragraph=False,
            detail=1,
            batch_size=1
        )

        if not results:
            return "", 0.0

        texts = []
        confidences = []

        for detection in results:
            if len(detection) == 3:
                bbox, text, conf = detection
            elif len(detection) == 2:
                text, conf = detection
            else:
                continue

            texts.append(text.strip())
            confidences.append(conf)

        combined_text = ' '.join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return combined_text, avg_confidence

    except Exception as e:
        print(f"OCR Error: {e}")
        return "", 0.0


# ========== Helper Functions ==========

def process_single_screenshot(image_path: str) -> List[Dict[str, Any]]:
    """
    단일 스크린샷에서 메시지 추출 (main.py 로직 재사용)

    Args:
        image_path: 이미지 경로

    Returns:
        추출된 메시지 리스트
    """
    global ocr_reader

    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # OCR 리더 가져오기
    if ocr_reader is None:
        ocr_reader = get_ocr_reader()

    # 말풍선 감지
    bubbles, debug_binary = detect_chat_bubbles(image)
    print(f"  감지된 말풍선: {len(bubbles)}개")

    # 텍스트 추출
    messages = []

    for idx, bubble in enumerate(bubbles, 1):
        text, confidence = extract_text_from_roi(image, bubble, ocr_reader)

        if not text:
            continue

        # 필터링
        if is_ui_element_or_noise(text, bubble):
            continue

        if is_repeated_sender_name(text, bubble, messages):
            continue

        # speaker 변환
        speaker = 'user' if bubble['bubble_type'] == 'right' else 'interlocutor'

        message_data = {
            'text': text,
            'confidence': round(confidence, 3),
            'speaker': speaker,
            'position': {
                'x': float(bubble['x']),
                'y': float(bubble['y']),
                'width': float(bubble['width']),
                'height': float(bubble['height'])
            }
        }
        messages.append(message_data)

    # 후처리: 반복되는 발신자 이름 제거
    interlocutor_texts = [
        msg['text'] for msg in messages
        if msg['speaker'] == 'interlocutor' and len(msg['text'].strip()) <= 5
    ]
    text_counts = {text: interlocutor_texts.count(text) for text in set(interlocutor_texts)}
    names_to_filter = {text for text, count in text_counts.items() if count > 1}

    if names_to_filter:
        messages = [
            msg for msg in messages
            if not (msg['speaker'] == 'interlocutor' and msg['text'] in names_to_filter)
        ]

    return messages


# ========== API Endpoints ==========

@app.get("/")
async def root():
    """API 루트"""
    return {
        "message": "Chat OCR API V2 - Session-based",
        "version": "2.0.0",
        "endpoints": {
            "POST /sessions": "새 세션 생성",
            "POST /sessions/{session_id}/upload": "스크린샷 업로드",
            "POST /sessions/{session_id}/process": "세션 처리 (병합 + 외부 API)",
            "GET /sessions/{session_id}/messages": "메시지 조회",
            "POST /sessions/{session_id}/search": "스크린샷으로 메시지 검색",
            "POST /sessions/{session_id}/view": "스크린샷으로 분석 결과 조회 (fuzzy matching)"
        }
    }


@app.post("/sessions", response_model=SessionCreateResponse)
async def create_session():
    """새 세션 생성"""
    session_id = str(uuid.uuid4())

    try:
        session = db.create_session(session_id)
        print(f"✓ 세션 생성: {session_id}")

        return SessionCreateResponse(
            session_id=session['session_id'],
            created_at=session['created_at'],
            status=session['status']
        )

    except Exception as e:
        print(f"세션 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@app.post("/sessions/{session_id}/upload", response_model=ScreenshotUploadResponse)
async def upload_screenshot(
    session_id: str = FastAPIPath(..., description="세션 ID"),
    file: UploadFile = File(...)
):
    """
    세션에 스크린샷 업로드

    Args:
        session_id: 세션 ID
        file: 이미지 파일

    Returns:
        업로드 결과
    """
    # 세션 확인
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # 파일 검증
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        # 업로드 순서 계산
        existing_screenshots = db.get_screenshots(session_id)
        upload_order = len(existing_screenshots) + 1

        # 파일 저장
        screenshot_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        filename = f"{session_id}_{upload_order}_{screenshot_id[:8]}{file_extension}"
        file_path = UPLOAD_DIR / filename

        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)

        # 이미지 크기 확인
        image = cv2.imread(str(file_path))
        if image is None:
            raise ValueError("Failed to load image")

        img_height, img_width = image.shape[:2]

        # DB에 저장
        screenshot = db.add_screenshot(
            screenshot_id=screenshot_id,
            session_id=session_id,
            file_path=str(file_path),
            upload_order=upload_order,
            image_width=img_width,
            image_height=img_height
        )

        print(f"✓ 스크린샷 업로드: {filename} (순서: {upload_order})")

        return ScreenshotUploadResponse(
            screenshot_id=screenshot_id,
            session_id=session_id,
            upload_order=upload_order,
            processed=False,
            message=f"Screenshot uploaded successfully (order: {upload_order})"
        )

    except Exception as e:
        print(f"업로드 실패: {e}")
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/sessions/{session_id}/process", response_model=ProcessSessionResponse)
async def process_session(
    session_id: str = FastAPIPath(..., description="세션 ID"),
    relationship: str = Query(..., description="대화 상대와의 관계"),
    relationship_info: str = Query(..., description="관계에 대한 추가 정보")
):
    """
    세션 처리: OCR → 병합 → 외부 API 호출

    Args:
        session_id: 세션 ID
        relationship: 대화 상대와의 관계 (예: "FRIEND", "SUPERIOR" 등)
        relationship_info: 관계에 대한 추가 정보 (예: "2년 지기", "신입사원" 등)

    Returns:
        처리 결과
    """
    # 세션 확인
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        print(f"\n{'='*70}")
        print(f"세션 처리 시작: {session_id}")
        print(f"{'='*70}")

        # 0. relationship 정보 저장
        db.update_session_relationship(session_id, relationship, relationship_info)
        print(f"  대화 상대: {relationship} ({relationship_info})")

        # 1. 스크린샷 가져오기
        screenshots = db.get_screenshots(session_id)
        if not screenshots:
            raise HTTPException(status_code=400, detail="No screenshots uploaded")

        print(f"\n📸 총 {len(screenshots)}개 스크린샷")

        # 2. 각 스크린샷에서 OCR 수행
        all_screenshot_messages = []

        for idx, screenshot in enumerate(screenshots, 1):
            print(f"\n[{idx}/{len(screenshots)}] OCR 처리: {Path(screenshot['file_path']).name}")

            messages = process_single_screenshot(screenshot['file_path'])
            print(f"  추출된 메시지: {len(messages)}개")

            # 메시지에 screenshot_id 추가
            for msg in messages:
                msg['screenshot_id'] = screenshot['screenshot_id']

            all_screenshot_messages.append(messages)
            db.mark_screenshot_processed(screenshot['screenshot_id'])

        # 3. 스크린샷 병합
        print(f"\n{'='*70}")
        print("스크린샷 병합 시작")
        print(f"{'='*70}")

        merged_messages, merge_history = merge_multiple_screenshots(
            all_screenshot_messages,
            min_overlap=2
        )

        print(f"\n병합 결과: {len(merged_messages)}개 메시지")

        # 4. 중복 제거
        merged_messages = deduplicate_messages(merged_messages)

        # 5. 그룹 ID 재할당
        merged_messages = assign_global_group_ids(merged_messages)

        # 6. DB에 저장
        print(f"\n💾 데이터베이스에 저장 중...")
        for i, msg in enumerate(merged_messages):
            message_id = str(uuid.uuid4())
            msg['message_id'] = message_id

            db.add_message(
                message_id=message_id,
                session_id=session_id,
                screenshot_id=msg['screenshot_id'],
                text=msg['text'],
                speaker=msg['speaker'],
                confidence=msg['confidence'],
                position_x=msg['position']['x'],
                position_y=msg['position']['y'],
                position_width=msg['position']['width'],
                position_height=msg['position']['height'],
                group_id=msg.get('group_id'),
                sequence_order=i
            )

        # 7. 외부 API 호출 (user 메시지에 score/ai_message 추가)
        print(f"\n{'='*70}")
        print("외부 서버 연동")
        print(f"{'='*70}")

        external_service = get_external_service(
            api_url=EXTERNAL_API_URL,
            api_key=EXTERNAL_API_KEY
        )
        score_results = await external_service.get_scores_for_messages(
            merged_messages,
            relationship=relationship,
            relationship_info=relationship_info
        )

        if score_results:
            # API 응답을 DB 스키마에 맞게 변환
            transformed_results = []
            for res in score_results:
                transformed_results.append({
                    'group_id': res.get('group_id'),
                    'score': res.get('appropriateness_rating'),  # appropriateness_rating을 score로 매핑
                    'emotional_tone': res.get('emotional_tone'),
                    'impact_score': res.get('impact_score'),
                    'review_comment': res.get('review_comment'),
                    'suggested_alternative': res.get('suggested_alternative'),
                })
            
            db.bulk_update_scores_by_group(session_id, transformed_results)
            print(f"✓ {len(transformed_results)}개 그룹에 score 업데이트 완료")

        # 8. 세션 상태 업데이트
        db.update_session_counts(session_id)
        db.update_session_status(session_id, 'completed')

        print(f"\n{'='*70}")
        print(f"✓ 세션 처리 완료: {session_id}")
        print(f"{'='*70}\n")

        return ProcessSessionResponse(
            session_id=session_id,
            status='completed',
            total_screenshots=len(screenshots),
            total_messages=len(merged_messages),
            merge_info={
                'merge_history': merge_history,
                'total_merged': len(merged_messages)
            },
            external_api_called=len(score_results) > 0
        )

    except Exception as e:
        print(f"\n❌ 세션 처리 실패: {e}")
        db.update_session_status(session_id, 'failed')
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/sessions/{session_id}/messages", response_model=SessionMessagesResponse)
async def get_session_messages(session_id: str = FastAPIPath(..., description="세션 ID")):
    """
    세션의 메시지 조회

    Args:
        session_id: 세션 ID

    Returns:
        메시지 리스트
    """
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        messages = db.get_messages(session_id, order_by='sequence_order')
        screenshots = db.get_screenshots(session_id)

        message_models = [
            MessageModel(
                message_id=msg['message_id'],
                text=msg['text'],
                speaker=msg['speaker'],
                confidence=msg['confidence'],
                position=msg['position'],
                group_id=msg.get('group_id'),
                score=msg.get('score'),
                emotional_tone=msg.get('emotional_tone'),
                impact_score=msg.get('impact_score'),
                ai_message=msg.get('review_comment')  # review_comment를 ai_message로 매핑
            )
            for msg in messages
        ]

        return SessionMessagesResponse(
            session_id=session_id,
            total_messages=len(messages),
            total_screenshots=len(screenshots),
            messages=message_models
        )

    except Exception as e:
        print(f"메시지 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get messages: {str(e)}")


@app.post("/sessions/{session_id}/search")
async def search_by_screenshot(
    session_id: str = FastAPIPath(..., description="세션 ID"),
    file: UploadFile = File(...)
):
    """
    스크린샷으로 메시지 검색 (OCR 없이 기존 데이터에서 찾기)

    Args:
        session_id: 세션 ID
        file: 검색용 스크린샷

    Returns:
        매칭된 메시지 리스트
    """
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Session not processed yet")

    try:
        # 임시 파일 저장
        temp_path = UPLOAD_DIR / f"search_{uuid.uuid4()}{Path(file.filename).suffix}"
        async with aiofiles.open(temp_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)

        # 간단한 OCR로 텍스트 추출
        print(f"🔍 검색용 스크린샷 분석 중...")
        search_messages = process_single_screenshot(str(temp_path))

        # 임시 파일 삭제
        temp_path.unlink()

        if not search_messages:
            return JSONResponse(content={
                "matched": False,
                "message": "No messages found in search screenshot",
                "results": []
            })

        # DB에서 매칭되는 메시지 찾기
        all_messages = db.get_messages(session_id)
        matched_messages = []

        print(f"  검색 메시지: {len(search_messages)}개")
        print(f"  세션 메시지: {len(all_messages)}개")

        # 간단한 텍스트 매칭
        for search_msg in search_messages:
            for db_msg in all_messages:
                if search_msg['text'] == db_msg['text'] and search_msg['speaker'] == db_msg['speaker']:
                    matched_messages.append(db_msg)
                    break

        print(f"  ✓ 매칭된 메시지: {len(matched_messages)}개")

        return JSONResponse(content={
            "matched": len(matched_messages) > 0,
            "message": f"Found {len(matched_messages)} matching messages",
            "results": matched_messages
        })

    except Exception as e:
        print(f"검색 실패: {e}")
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/sessions/{session_id}/view")
async def view_by_screenshots(
    session_id: str = FastAPIPath(..., description="세션 ID"),
    files: List[UploadFile] = File(..., description="조회용 스크린샷들")
):
    """
    스크린샷으로 이미 분석된 메시지 조회 (Fuzzy matching 지원)

    - 여러 스크린샷 업로드 가능
    - OCR 수행 후 DB의 기존 분석 결과와 fuzzy matching
    - 매칭된 메시지만 반환 (AI 분석 결과 포함)
    - 매칭 안된 메시지는 제외

    Args:
        session_id: 세션 ID
        files: 조회용 스크린샷 파일들

    Returns:
        매칭된 메시지 리스트 (AI 분석 결과 포함)
    """
    from difflib import SequenceMatcher

    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Session not processed yet. Please call /process first.")

    try:
        print(f"\n{'='*70}")
        print(f"View 요청: {session_id}")
        print(f"{'='*70}")
        print(f"📸 업로드된 스크린샷: {len(files)}개")

        # 1. 임시 파일 저장 및 OCR 처리
        temp_paths = []
        all_view_messages = []

        for idx, file in enumerate(files, 1):
            # 임시 파일 저장
            temp_path = UPLOAD_DIR / f"view_{uuid.uuid4()}{Path(file.filename).suffix}"
            async with aiofiles.open(temp_path, 'wb') as out_file:
                content = await file.read()
                await out_file.write(content)
            temp_paths.append(temp_path)

            # OCR 처리
            print(f"\n[{idx}/{len(files)}] OCR 처리: {file.filename}")
            messages = process_single_screenshot(str(temp_path))
            print(f"  추출된 메시지: {len(messages)}개")
            all_view_messages.extend(messages)

        print(f"\n총 OCR 추출 메시지: {len(all_view_messages)}개")

        # 2. DB에서 기존 분석된 메시지 가져오기
        db_messages = db.get_messages(session_id, order_by='sequence_order')
        print(f"DB 저장된 메시지: {len(db_messages)}개")

        # 3. Fuzzy matching으로 매칭
        def text_similarity(text1: str, text2: str) -> float:
            """텍스트 유사도 계산 (0.0 ~ 1.0)"""
            return SequenceMatcher(None, text1, text2).ratio()

        matched_results = []
        SIMILARITY_THRESHOLD = 0.85  # 85% 이상 유사하면 매칭

        for view_msg in all_view_messages:
            best_match = None
            best_score = 0.0

            for db_msg in db_messages:
                # speaker 일치 확인
                if view_msg['speaker'] != db_msg['speaker']:
                    continue

                # 텍스트 유사도 계산
                similarity = text_similarity(view_msg['text'], db_msg['text'])

                if similarity > best_score:
                    best_score = similarity
                    best_match = db_msg

            # 임계값 이상이면 매칭 성공
            if best_match and best_score >= SIMILARITY_THRESHOLD:
                # 중복 제거 (이미 추가된 message_id는 스킵)
                if not any(m['message_id'] == best_match['message_id'] for m in matched_results):
                    matched_results.append(best_match)
                    print(f"  ✓ 매칭: '{view_msg['text'][:30]}...' → '{best_match['text'][:30]}...' (유사도: {best_score:.2f})")

        # 4. 임시 파일 삭제
        for temp_path in temp_paths:
            if temp_path.exists():
                temp_path.unlink()

        # 5. 결과 정렬 (sequence_order 기준)
        matched_results.sort(key=lambda x: x.get('sequence_order', 0))

        print(f"\n✓ 최종 매칭된 메시지: {len(matched_results)}개")
        print(f"{'='*70}\n")

        # 6. Response 생성
        message_models = [
            {
                'message_id': msg['message_id'],
                'text': msg['text'],
                'speaker': msg['speaker'],
                'confidence': msg['confidence'],
                'position': msg['position'],
                'group_id': msg.get('group_id'),
                'score': msg.get('score'),
                'emotional_tone': msg.get('emotional_tone'),
                'impact_score': msg.get('impact_score'),
                'ai_message': msg.get('review_comment'),
                'suggested_alternative': msg.get('suggested_alternative')
            }
            for msg in matched_results
        ]

        return JSONResponse(content={
            "session_id": session_id,
            "matched": len(matched_results) > 0,
            "total_matched": len(matched_results),
            "total_ocr_extracted": len(all_view_messages),
            "messages": message_models
        })

    except Exception as e:
        print(f"\n❌ View 실패: {e}")
        # 임시 파일 정리
        for temp_path in temp_paths:
            if temp_path.exists():
                temp_path.unlink()
        raise HTTPException(status_code=500, detail=f"View failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=80,
        reload=True
    )
