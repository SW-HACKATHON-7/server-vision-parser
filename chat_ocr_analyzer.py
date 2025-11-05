"""
카카오톡 채팅 말풍선 OCR 분석기
이미지에서 말풍선(채팅 박스)을 감지하고 OCR을 수행하여 JSON으로 반환
"""

import cv2
import numpy as np
import easyocr
import json
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ChatBubble:
    """채팅 말풍선 데이터 클래스"""
    id: int
    text: str
    position: Dict[str, float]  # x, y, width, height
    confidence: float
    bubble_type: str  # 'left' (받은 메시지) or 'right' (보낸 메시지)
    timestamp: str = ""
    sender: str = ""


class ChatOCRAnalyzer:
    """채팅 이미지 OCR 분석기"""

    def __init__(self, languages=['ko', 'en']):
        """
        Args:
            languages: OCR에 사용할 언어 리스트
        """
        print("Initializing EasyOCR Reader...")
        self.reader = easyocr.Reader(languages, gpu=True)
        print("EasyOCR Reader initialized successfully")

    def detect_chat_bubbles(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        이미지에서 채팅 말풍선 영역을 감지

        Args:
            image: 입력 이미지 (BGR)

        Returns:
            감지된 말풍선 영역 리스트
        """
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 이진화
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        # 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 윤곽선 찾기
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bubbles = []
        img_height, img_width = image.shape[:2]

        for contour in contours:
            # 바운딩 박스 얻기
            x, y, w, h = cv2.boundingRect(contour)

            # 너무 작거나 큰 영역 필터링
            if w < 50 or h < 20 or w > img_width * 0.9 or h > img_height * 0.3:
                continue

            # 종횡비 체크 (말풍선은 보통 가로로 긴 형태)
            aspect_ratio = w / h
            if aspect_ratio < 0.5 or aspect_ratio > 10:
                continue

            # 말풍선 타입 결정 (이미지의 좌/우 위치로 판단)
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

        return bubbles

    def extract_text_from_bubble(self, image: np.ndarray, bubble: Dict[str, Any]) -> tuple:
        """
        말풍선 영역에서 텍스트 추출

        Args:
            image: 원본 이미지
            bubble: 말풍선 영역 정보

        Returns:
            (추출된 텍스트, 신뢰도)
        """
        x, y, w, h = bubble['x'], bubble['y'], bubble['width'], bubble['height']

        # ROI 추출 (여백 추가)
        padding = 5
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)

        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            return "", 0.0

        # OCR 수행
        try:
            results = self.reader.readtext(roi, paragraph=True)

            if not results:
                return "", 0.0

            # 모든 텍스트 결합
            texts = []
            confidences = []

            for detection in results:
                bbox, text, conf = detection
                texts.append(text.strip())
                confidences.append(conf)

            combined_text = ' '.join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return combined_text, avg_confidence

        except Exception as e:
            print(f"OCR error: {e}")
            return "", 0.0

    def analyze_chat_image(self, image_path: str) -> Dict[str, Any]:
        """
        채팅 이미지를 분석하여 JSON 형태로 반환

        Args:
            image_path: 이미지 파일 경로

        Returns:
            분석 결과 딕셔너리
        """
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        print(f"Analyzing image: {image_path}")
        print(f"Image size: {image.shape[1]}x{image.shape[0]}")

        # 말풍선 감지
        bubbles = self.detect_chat_bubbles(image)
        print(f"Detected {len(bubbles)} chat bubbles")

        # 각 말풍선에서 텍스트 추출
        chat_messages = []

        for idx, bubble in enumerate(bubbles, 1):
            text, confidence = self.extract_text_from_bubble(image, bubble)

            if text:  # 텍스트가 있는 경우만 추가
                chat_bubble = ChatBubble(
                    id=idx,
                    text=text,
                    position={
                        'x': float(bubble['x']),
                        'y': float(bubble['y']),
                        'width': float(bubble['width']),
                        'height': float(bubble['height'])
                    },
                    confidence=round(confidence, 3),
                    bubble_type=bubble['bubble_type']
                )
                chat_messages.append(asdict(chat_bubble))
                print(f"  [{idx}] {bubble['bubble_type']}: {text[:50]}...")

        # 결과 구성
        result = {
            'image_path': image_path,
            'image_size': {
                'width': image.shape[1],
                'height': image.shape[0]
            },
            'total_messages': len(chat_messages),
            'messages': chat_messages
        }

        return result

    def save_result_with_visualization(self, image_path: str, result: Dict[str, Any],
                                      output_dir: str = 'results'):
        """
        분석 결과를 JSON으로 저장하고 시각화 이미지 생성

        Args:
            image_path: 원본 이미지 경로
            result: 분석 결과
            output_dir: 결과 저장 디렉토리
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # JSON 저장
        json_file = output_path / 'chat_analysis.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"JSON saved to: {json_file}")

        # 시각화 이미지 생성
        image = cv2.imread(image_path)

        for msg in result['messages']:
            pos = msg['position']
            x, y, w, h = int(pos['x']), int(pos['y']), int(pos['width']), int(pos['height'])

            # 바운딩 박스 색상 (받은 메시지: 파랑, 보낸 메시지: 초록)
            color = (255, 0, 0) if msg['bubble_type'] == 'left' else (0, 255, 0)

            # 박스 그리기
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            # ID 표시
            label = f"#{msg['id']}"
            cv2.putText(image, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 이미지 저장
        vis_file = output_path / 'chat_visualization.jpg'
        cv2.imwrite(str(vis_file), image)
        print(f"Visualization saved to: {vis_file}")


def main():
    """메인 실행 함수"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python chat_ocr_analyzer.py <image_path>")
        print("Example: python chat_ocr_analyzer.py chat_screenshot.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)

    # 분석기 초기화
    analyzer = ChatOCRAnalyzer(languages=['ko', 'en'])

    # 이미지 분석
    result = analyzer.analyze_chat_image(image_path)

    # 결과 저장
    analyzer.save_result_with_visualization(image_path, result)

    # 결과 출력
    print("\n" + "="*50)
    print("Analysis Result:")
    print("="*50)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
