"""
독립 실행형 채팅 OCR 스크립트
FastAPI 서버 없이 target.png 파일을 직접 처리
"""

import cv2
import numpy as np
import easyocr
import json
from pathlib import Path
from typing import List, Dict, Any


class SimpleChatOCR:
    """간단한 채팅 OCR 프로세서"""

    def __init__(self, languages=['ko', 'en']):
        """
        Args:
            languages: OCR 언어 리스트
        """
        print("EasyOCR 초기화 중... (처음 실행 시 모델 다운로드로 1~2분 소요)")
        import time
        start = time.time()
        self.reader = easyocr.Reader(languages, gpu=True)
        elapsed = time.time() - start
        print(f"초기화 완료! ({elapsed:.1f}초 소요)")

    def detect_chat_bubbles(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        채팅 말풍선 감지 (라이트/다크 모드 자동 감지)

        Args:
            image: OpenCV 이미지 (BGR)

        Returns:
            말풍선 리스트
        """
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 배경 밝기 판단 (평균 밝기)
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

        bubbles = []
        img_height, img_width = image.shape[:2]

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # 필터링 (더 관대한 조건)
            if w < 40 or h < 15:
                continue
            if w > img_width * 0.95 or h > img_height * 0.4:
                continue

            # 면적 체크
            area = cv2.contourArea(contour)
            if area < 500:  # 너무 작은 영역 제외
                continue

            aspect_ratio = w / h
            if aspect_ratio < 0.3 or aspect_ratio > 15:
                continue

            # 말풍선 타입 결정
            center_x = x + w / 2
            bubble_type = 'right' if center_x > img_width * 0.5 else 'left'

            bubbles.append({
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'bubble_type': bubble_type
            })

        # y 좌표 정렬
        bubbles.sort(key=lambda b: b['y'])

        print(f"  - 다크모드: {'예' if is_dark_mode else '아니오'} (평균 밝기: {avg_brightness:.1f})")

        return bubbles, binary

    def extract_text(self, image: np.ndarray, bubble: Dict[str, Any]) -> tuple:
        """
        ROI에서 텍스트 추출

        Args:
            image: 원본 이미지
            bubble: 말풍선 정보

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
            results = self.reader.readtext(roi)

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
            print(f"OCR 에러: {e}")
            return "", 0.0

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        이미지 처리

        Args:
            image_path: 이미지 파일 경로

        Returns:
            분석 결과 딕셔너리
        """
        print(f"\n이미지 로드 중: {image_path}")

        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")

        img_height, img_width = image.shape[:2]
        print(f"이미지 크기: {img_width}x{img_height}")

        # 말풍선 감지
        print("\n말풍선 감지 중...")
        bubbles, debug_binary = self.detect_chat_bubbles(image)
        print(f"감지된 말풍선: {len(bubbles)}개")

        # 텍스트 추출
        print(f"\nOCR 수행 중... (총 {len(bubbles)}개 말풍선)")
        messages = []

        import time
        start_ocr = time.time()

        for idx, bubble in enumerate(bubbles, 1):
            bubble_start = time.time()
            text, confidence = self.extract_text(image, bubble)

            if text:
                message = {
                    'id': idx,
                    'text': text,
                    'confidence': round(confidence, 3),
                    'bubble_type': bubble['bubble_type'],
                    'position': {
                        'x': float(bubble['x']),
                        'y': float(bubble['y']),
                        'width': float(bubble['width']),
                        'height': float(bubble['height'])
                    }
                }
                messages.append(message)

                # 진행 상황 출력
                icon = "[받음]" if bubble['bubble_type'] == 'left' else "[보냄]"
                elapsed = time.time() - bubble_start
                print(f"  [{idx}/{len(bubbles)}] {icon} {text[:50]}{'...' if len(text) > 50 else ''} ({elapsed:.1f}초)")

        total_ocr_time = time.time() - start_ocr
        print(f"\nOCR 완료! 총 소요 시간: {total_ocr_time:.1f}초 (평균 {total_ocr_time/len(bubbles):.1f}초/말풍선)")

        # 결과 구성
        result = {
            'image_path': image_path,
            'image_size': {
                'width': img_width,
                'height': img_height
            },
            'total_messages': len(messages),
            'messages': messages,
            'debug_binary': debug_binary  # 디버그용
        }

        return result

    def save_results(self, image_path: str, result: Dict[str, Any], output_dir: str = 'output'):
        """
        결과 저장 (JSON + 시각화 이미지)

        Args:
            image_path: 원본 이미지 경로
            result: 분석 결과
            output_dir: 출력 디렉토리
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # 디버그 이미지 추출 (JSON에 저장하지 않음)
        debug_binary = result.pop('debug_binary', None)

        # JSON 저장
        json_file = output_path / 'result.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n저장 OK - JSON: {json_file}")

        # 디버그 이진화 이미지 저장
        if debug_binary is not None:
            debug_file = output_path / 'debug_binary.jpg'
            cv2.imwrite(str(debug_file), debug_binary)
            print(f"저장 OK - 디버그 이미지: {debug_file}")

        # 시각화 이미지 생성
        image = cv2.imread(image_path)

        for msg in result['messages']:
            pos = msg['position']
            x, y = int(pos['x']), int(pos['y'])
            w, h = int(pos['width']), int(pos['height'])

            # 색상: 받은 메시지(파랑), 보낸 메시지(초록)
            color = (255, 0, 0) if msg['bubble_type'] == 'left' else (0, 255, 0)

            # 바운딩 박스
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            # ID 표시
            label = f"#{msg['id']}"
            cv2.putText(image, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 이미지 저장
        vis_file = output_path / 'visualization.jpg'
        cv2.imwrite(str(vis_file), image)
        print(f"저장 OK - 시각화 이미지: {vis_file}")

        print(f"\n모든 결과가 저장되었습니다: {output_path}/")


def main():
    """메인 함수"""
    import sys

    # 기본값: target.png
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "target.png"

    # 파일 존재 확인
    if not Path(image_path).exists():
        print(f"에러 - 파일을 찾을 수 없습니다: {image_path}")
        print(f"\n사용법:")
        print(f"  python simple_chat_ocr.py              # target.png 처리")
        print(f"  python simple_chat_ocr.py <image_path>  # 다른 이미지 처리")
        sys.exit(1)

    print("="*70)
    print("        카카오톡 채팅 OCR 분석기")
    print("="*70)

    try:
        # OCR 프로세서 초기화
        ocr = SimpleChatOCR(languages=['ko', 'en'])

        # 이미지 처리
        result = ocr.process_image(image_path)

        # 결과 출력
        print("\n" + "="*70)
        print("분석 결과 요약")
        print("="*70)
        print(f"총 메시지 수: {result['total_messages']}")
        print(f"받은 메시지: {sum(1 for m in result['messages'] if m['bubble_type'] == 'left')}개")
        print(f"보낸 메시지: {sum(1 for m in result['messages'] if m['bubble_type'] == 'right')}개")

        # 결과 저장
        ocr.save_results(image_path, result)

        print("\n처리 OK -")

    except Exception as e:
        print(f"\n에러 -: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
