"""
FastAPI 기반 카카오톡 채팅 말풍선 OCR 분석 API
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import cv2
import numpy as np
import easyocr
import json
import uuid
from datetime import datetime
from pathlib import Path
import aiofiles
import shutil


# Pydantic 모델 정의
class Position(BaseModel):
    """위치 정보 모델"""
    x: float = Field(..., description="X 좌표")
    y: float = Field(..., description="Y 좌표")
    width: float = Field(..., description="너비")
    height: float = Field(..., description="높이")


class ChatMessage(BaseModel):
    """채팅 메시지 모델"""
    id: int = Field(..., description="메시지 ID")
    text: str = Field(..., description="추출된 텍스트")
    confidence: float = Field(..., description="OCR 신뢰도 (0-1)")
    speaker: str = Field(..., description="발화자 (user: 사용자, interlocutor: 대화상대)")
    position: Position = Field(..., description="말풍선 위치")
    group_id: int = Field(..., description="같은 speaker의 연속 메시지 그룹 ID")


class ChatAnalysisResult(BaseModel):
    """채팅 분석 결과 모델"""
    analysis_id: str = Field(..., description="분석 ID")
    image_path: str = Field(..., description="업로드된 이미지 경로")
    analyzed_at: str = Field(..., description="분석 시각")
    total_messages: int = Field(..., description="총 메시지 수")
    image_size: Dict[str, int] = Field(..., description="이미지 크기 (width, height)")
    messages: List[ChatMessage] = Field(..., description="추출된 메시지 리스트")


class AnalysisResponse(BaseModel):
    """API 응답 모델"""
    status: str = Field(..., description="응답 상태 (success/error)")
    message: str = Field(..., description="응답 메시지")
    data: Optional[ChatAnalysisResult] = Field(None, description="분석 결과 데이터")


# FastAPI 앱 생성
app = FastAPI(
    title="Chat OCR API",
    description="카카오톡 채팅 말풍선 이미지를 분석하여 텍스트를 추출하는 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
ocr_reader = None
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")

# 디렉토리 생성
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def get_ocr_reader():
    """OCR 리더 싱글톤"""
    global ocr_reader
    if ocr_reader is None:
        print("Initializing EasyOCR Reader...")
        ocr_reader = easyocr.Reader(['ko', 'en'], gpu=True)
        print("EasyOCR Reader initialized successfully")
    return ocr_reader


def is_ui_element_or_noise(text: str, bubble: Dict[str, Any]) -> bool:
    """
    UI 요소나 노이즈 텍스트 필터링

    Args:
        text: OCR 텍스트
        bubble: 말풍선 정보

    Returns:
        필터링 대상이면 True
    """
    import re

    text_clean = text.strip()

    # UI 요소
    ui_keywords = ['TALK', '메시지 입력', 'LTE', '검색', '설정']
    if any(keyword in text_clean for keyword in ui_keywords):
        return True

    # 순수 시간만 있는 경우 (짧고 시간 패턴만)
    # 예: "오후 7:42", "95 7:43" 등
    if len(text_clean) < 15:
        # 시간 패턴이 전체 텍스트의 대부분을 차지하는 경우
        time_match = re.search(r'(오전|오후|AM|PM)?\s*\d{1,2}[:\.]?\d{2}', text_clean)
        if time_match:
            # 시간 부분을 제거했을 때 남는 텍스트가 거의 없으면 필터링
            remaining = text_clean.replace(time_match.group(), '').strip()
            if len(remaining) <= 3:  # 3글자 이하만 남으면 시간 전용
                return True

    # 너무 짧은 텍스트 (1-2글자)
    if len(text_clean) <= 2:
        return True

    # 날짜 패턴
    if re.match(r'\d{4}년\s+\d{1,2}월\s+\d{1,2}일', text_clean):
        return True

    return False


def is_repeated_sender_name(text: str, bubble: Dict[str, Any], previous_messages: List[Dict]) -> bool:
    """
    반복되는 발신자 이름 필터링

    Args:
        text: 현재 텍스트
        bubble: 현재 말풍선
        previous_messages: 이전 메시지 리스트

    Returns:
        필터링 대상이면 True
    """
    # 왼쪽(받은 메시지)이고, 크기가 작고, 짧은 텍스트
    if bubble['bubble_type'] == 'left' and bubble['width'] < 80 and len(text) <= 4:
        # 최근 3개 메시지에서 같은 텍스트 반복 체크
        recent_texts = [msg['text'] for msg in previous_messages[-3:]]
        if recent_texts.count(text) >= 2:
            return True

        # 발신자 이름으로 보이는 패턴 (짧고 왼쪽 상단)
        if bubble['height'] < 30 and len(text) <= 5:
            return True

    return False


def detect_chat_bubbles(image: np.ndarray) -> List[Dict[str, Any]]:
    """
    이미지에서 채팅 말풍선 영역 감지 (라이트/다크 모드 자동 감지)

    Args:
        image: OpenCV 이미지 (BGR)

    Returns:
        감지된 말풍선 리스트
    """
    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 배경 밝기 판단
    avg_brightness = np.mean(gray)
    is_dark_mode = avg_brightness < 100

    if is_dark_mode:
        # 다크모드: 밝은 말풍선 찾기
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    else:
        # 라이트모드: 어두운 말풍선 찾기
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # 윤곽선 찾기
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"  - 총 {len(contours)}개 윤곽선 발견")

    bubbles = []
    img_height, img_width = image.shape[:2]
    filtered_count = {'size': 0, 'area': 0, 'aspect': 0}

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # 필터링 (더 관대한 조건)
        if w < 40 or h < 15:
            filtered_count['size'] += 1
            continue
        if w > img_width * 0.95 or h > img_height * 0.4:
            filtered_count['size'] += 1
            continue

        # 면적 체크
        area = cv2.contourArea(contour)
        if area < 500:
            filtered_count['area'] += 1
            continue

        # 종횡비 체크
        aspect_ratio = w / h
        if aspect_ratio < 0.3 or aspect_ratio > 15:
            filtered_count['aspect'] += 1
            continue

        # 말풍선 타입 결정 (좌측/우측 위치로 판단)
        center_x = x + w / 2
        bubble_type = 'right' if center_x > img_width * 0.5 else 'left'

        bubbles.append({
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'bubble_type': bubble_type
        })

    # y 좌표로 정렬 (위에서 아래로)
    bubbles.sort(key=lambda b: b['y'])

    print(f"  - 필터링됨: 크기({filtered_count['size']}), 면적({filtered_count['area']}), 종횡비({filtered_count['aspect']})")
    print(f"  - 병합 전 말풍선: {len(bubbles)}개")

    # 인접한 말풍선 병합 (같은 줄에 있는 작은 조각들)
    merged_bubbles = merge_nearby_bubbles(bubbles, img_width, img_height)
    print(f"  - 병합 후 말풍선: {len(merged_bubbles)}개")

    return merged_bubbles, binary  # 디버그용 이진화 이미지도 반환


def merge_nearby_bubbles(bubbles: List[Dict[str, Any]], img_width: int, img_height: int) -> List[Dict[str, Any]]:
    """
    Y 위치 기반으로 먼저 그룹핑 후 병합

    Args:
        bubbles: 말풍선 리스트
        img_width: 이미지 너비
        img_height: 이미지 높이

    Returns:
        병합된 말풍선 리스트
    """
    if not bubbles:
        return []

    # 1단계: 프로필 이미지 같은 정사각형 제외
    filtered = []
    for b in bubbles:
        aspect = b['width'] / b['height'] if b['height'] > 0 else 0
        # 정사각형에 가까운 것 제외 (프로필 이미지)
        if 0.8 < aspect < 1.2 and b['width'] > 60 and b['height'] > 60:
            continue
        filtered.append(b)

    bubbles = filtered

    # 2단계: Y 위치로 그룹핑 (같은 줄의 조각들)
    groups = []
    used = set()

    for i, bubble in enumerate(bubbles):
        if i in used:
            continue

        group = [bubble]
        used.add(i)
        bubble_y = bubble['y'] + bubble['height'] / 2

        for j, other in enumerate(bubbles):
            if j in used:
                continue

            other_y = other['y'] + other['height'] / 2

            # 같은 줄 판단 (Y 위치 차이 < 60px)
            if abs(bubble_y - other_y) < 60:
                group.append(other)
                used.add(j)

        groups.append(group)

    # 3단계: 각 그룹을 하나의 말풍선으로 병합
    merged = []

    for group in groups:
        if not group:
            continue

        # 전체 바운딩 박스 계산
        min_x = min(b['x'] for b in group)
        max_x = max(b['x'] + b['width'] for b in group)
        min_y = min(b['y'] for b in group)
        max_y = max(b['y'] + b['height'] for b in group)

        # 중심으로 speaker 판별
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
    """
    ROI 영역에서 텍스트 추출

    Args:
        image: 원본 이미지
        bubble: 말풍선 영역 정보
        reader: EasyOCR 리더

    Returns:
        (텍스트, 신뢰도)
    """
    x, y, w, h = bubble['x'], bubble['y'], bubble['width'], bubble['height']

    # 패딩 추가
    padding = 5
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + w + padding)
    y2 = min(image.shape[0], y + h + padding)

    roi = image[y1:y2, x1:x2]

    if roi.size == 0:
        return "", 0.0

    try:
        results = reader.readtext(roi)

        if not results:
            return "", 0.0

        # 텍스트 결합
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


def group_consecutive_messages(messages: List[Dict], gap_threshold: int = 50) -> List[Dict]:
    """
    같은 speaker의 연속 메시지를 그룹핑

    Args:
        messages: 메시지 리스트 (y 좌표로 정렬되어 있어야 함)
        gap_threshold: 같은 그룹으로 간주할 최대 간격 (px)

    Returns:
        group_id가 추가된 메시지 리스트
    """
    if not messages:
        return []

    grouped_messages = []
    current_group_id = 1

    for i, msg in enumerate(messages):
        if i == 0:
            # 첫 메시지
            msg['group_id'] = current_group_id
            grouped_messages.append(msg)
            continue

        prev_msg = grouped_messages[-1]

        # 같은 speaker인지 확인
        same_speaker = (msg['speaker'] == prev_msg['speaker'])

        # 이전 메시지의 bottom과 현재 메시지의 top 사이 간격 계산
        prev_bottom = prev_msg['position']['y'] + prev_msg['position']['height']
        current_top = msg['position']['y']
        gap = current_top - prev_bottom

        # 같은 speaker이고 간격이 threshold 이하면 같은 그룹
        if same_speaker and gap <= gap_threshold:
            msg['group_id'] = current_group_id
        else:
            # 새 그룹 시작
            current_group_id += 1
            msg['group_id'] = current_group_id

        grouped_messages.append(msg)

    return grouped_messages


def save_visualization(image_path: str, messages: List[Dict], output_path: Path):
    """
    분석 결과를 시각화한 이미지 저장

    Args:
        image_path: 원본 이미지 경로
        messages: 메시지 리스트
        output_path: 출력 경로
    """
    image = cv2.imread(image_path)

    for msg in messages:
        pos = msg['position']
        x, y = int(pos['x']), int(pos['y'])
        w, h = int(pos['width']), int(pos['height'])

        # 색상: 대화상대(파랑), 사용자(초록)
        color = (255, 0, 0) if msg['speaker'] == 'interlocutor' else (0, 255, 0)

        # 바운딩 박스
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        # ID 표시
        label = f"#{msg['id']}"
        cv2.putText(image, label, (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite(str(output_path), image)


# API 엔드포인트
@app.get("/")
async def root():
    """API 루트"""
    return {
        "message": "Chat OCR API",
        "version": "1.0.0",
        "endpoints": {
            "POST /analyze": "채팅 이미지 분석",
            "GET /health": "헬스 체크"
        }
    }


@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy"}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_chat_image(file: UploadFile = File(...)):
    """
    채팅 이미지 분석 엔드포인트

    Args:
        file: 업로드된 이미지 파일

    Returns:
        분석 결과 JSON
    """
    # 파일 검증
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image file."
        )

    # 고유 ID 생성
    analysis_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 파일 저장
    file_extension = Path(file.filename).suffix
    upload_filename = f"{timestamp}_{analysis_id[:8]}{file_extension}"
    upload_path = UPLOAD_DIR / upload_filename

    try:
        # 파일 저장
        async with aiofiles.open(upload_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)

        print(f"Uploaded file saved: {upload_path}")

        # 이미지 로드
        image = cv2.imread(str(upload_path))
        if image is None:
            raise ValueError("Failed to load image")

        img_height, img_width = image.shape[:2]
        print(f"Image size: {img_width}x{img_height}")

        # OCR 리더 가져오기
        reader = get_ocr_reader()

        # 말풍선 감지
        bubbles, debug_binary = detect_chat_bubbles(image)
        print(f"Detected {len(bubbles)} chat bubbles")

        # 텍스트 추출
        messages = []
        processing_log = []

        print("\n=== OCR 처리 시작 ===")
        processing_log.append("\n=== OCR 처리 상세 로그 ===\n")

        for idx, bubble in enumerate(bubbles, 1):
            bubble_log = []
            bubble_log.append(f"\n[Bubble {idx}/{len(bubbles)}]")
            bubble_log.append(f"  위치: x={bubble['x']}, y={bubble['y']}, w={bubble['width']}, h={bubble['height']}")
            bubble_log.append(f"  타입: {bubble['bubble_type']}")

            print(f"\n[Bubble {idx}/{len(bubbles)}]")
            print(f"  위치: x={bubble['x']}, y={bubble['y']}, w={bubble['width']}, h={bubble['height']}")
            print(f"  타입: {bubble['bubble_type']}")

            text, confidence = extract_text_from_roi(image, bubble, reader)

            if not text:
                bubble_log.append(f"  결과 없음")
                processing_log.extend(bubble_log)
                print(f"  결과 없음")
                continue

            bubble_log.append(f"  OCR: '{text}' (신뢰도: {confidence:.3f})")
            print(f"  OCR: '{text}' (신뢰도: {confidence:.3f})")

            # UI 요소/노이즈 필터링
            if is_ui_element_or_noise(text, bubble):
                bubble_log.append(f"  SKIP - UI/Noise 필터링")
                processing_log.extend(bubble_log)
                print(f"  SKIP - UI/Noise 필터링")
                continue

            # 반복되는 발신자 이름 필터링
            if is_repeated_sender_name(text, bubble, messages):
                bubble_log.append(f"  SKIP - 반복되는 발신자 이름")
                processing_log.extend(bubble_log)
                print(f"  SKIP - 반복되는 발신자 이름")
                continue

            # bubble_type을 speaker로 변환
            speaker = 'user' if bubble['bubble_type'] == 'right' else 'interlocutor'

            message_data = {
                'id': len(messages) + 1,
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
            bubble_log.append(f"  추가 - 메시지 #{len(messages)} (speaker: {speaker})")
            processing_log.extend(bubble_log)
            print(f"  추가 - 메시지 #{len(messages)} (speaker: {speaker})")

        print(f"\n=== OCR 완료: 총 {len(messages)}개 메시지 추출 ===")

        # 메시지 그룹핑 (같은 speaker의 연속 메시지)
        print("\n=== 메시지 그룹핑 중 ===")
        messages = group_consecutive_messages(messages, gap_threshold=50)

        # 그룹 정보 출력
        group_counts = {}
        for msg in messages:
            group_id = msg['group_id']
            group_counts[group_id] = group_counts.get(group_id, 0) + 1

        print(f"총 {len(group_counts)}개 그룹 생성됨")
        for group_id, count in sorted(group_counts.items()):
            if count > 1:
                print(f"  그룹 {group_id}: {count}개 메시지 (연속 버블)")

        # 결과 저장
        result_folder = RESULTS_DIR / f"{timestamp}_{analysis_id[:8]}"
        result_folder.mkdir(exist_ok=True)

        # 로그 파일 생성
        log_file = result_folder / 'analysis_log.txt'
        log_content = []
        log_content.append(f"=== 이미지 분석 로그 ===")
        log_content.append(f"분석 ID: {analysis_id}")
        log_content.append(f"이미지 경로: {upload_path}")
        log_content.append(f"이미지 크기: {img_width}x{img_height}")
        log_content.append(f"분석 시각: {datetime.now().isoformat()}")
        log_content.append(f"\n총 {len(bubbles)}개 말풍선 감지됨")
        log_content.append(f"최종 {len(messages)}개 메시지 추출됨\n")

        # OCR 처리 로그 추가
        log_content.extend(processing_log)

        # 로그 파일 저장
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(log_content))

        print(f"Analysis log saved to: {log_file}")

        # 분석 결과 객체 생성
        result_data = ChatAnalysisResult(
            analysis_id=analysis_id,
            image_path=str(upload_path),
            analyzed_at=datetime.now().isoformat(),
            total_messages=len(messages),
            image_size={'width': img_width, 'height': img_height},
            messages=[ChatMessage(**msg) for msg in messages]
        )

        # JSON 저장
        json_path = result_folder / 'chat_analysis.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_data.model_dump(), f, ensure_ascii=False, indent=2)

        print(f"Results saved to: {json_path}")

        # 시각화 이미지 저장
        vis_path = result_folder / 'visualization.jpg'
        save_visualization(str(upload_path), messages, vis_path)

        # 디버그 이미지 저장 (이진화)
        debug_path = result_folder / 'debug_binary.jpg'
        cv2.imwrite(str(debug_path), debug_binary)

        # 원본 이미지 복사
        original_path = result_folder / f'original{file_extension}'
        shutil.copy(upload_path, original_path)

        print(f"Debug binary image saved to: {debug_path}")

        # 응답 반환
        return AnalysisResponse(
            status="success",
            message=f"Successfully analyzed {len(messages)} chat messages",
            data=result_data
        )

    except Exception as e:
        # 에러 발생 시 업로드 파일 삭제
        if upload_path.exists():
            upload_path.unlink()

        print(f"Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
