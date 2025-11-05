"""
FastAPI 채팅 OCR API 테스트 클라이언트
"""

import json
from pathlib import Path
import sys
import urllib.request
import urllib.error
from urllib.parse import urlencode


def test_chat_ocr(image_path: str, api_url: str = "http://localhost:8000"):
    """
    채팅 OCR API 테스트

    Args:
        image_path: 테스트할 이미지 파일 경로
        api_url: API 서버 URL
    """
    # 파일 존재 확인
    if not Path(image_path).exists():
        print(f"Error: 파일을 찾을 수 없습니다: {image_path}")
        return

    print(f"이미지 업로드 중: {image_path}")
    print(f"API URL: {api_url}/analyze")

    try:
        # 멀티파트 폼 데이터 생성
        boundary = '----WebKitFormBoundary' + ''.join([str(i) for i in range(16)])

        with open(image_path, 'rb') as f:
            file_data = f.read()

        filename = Path(image_path).name

        body = (
            f'--{boundary}\r\n'
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f'Content-Type: image/png\r\n\r\n'
        ).encode('utf-8')
        body += file_data
        body += f'\r\n--{boundary}--\r\n'.encode('utf-8')

        # API 호출
        req = urllib.request.Request(
            f"{api_url}/analyze",
            data=body,
            headers={
                'Content-Type': f'multipart/form-data; boundary={boundary}'
            },
            method='POST'
        )

        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))

            print("\n" + "="*70)
            print("✓ 분석 성공!")
            print("="*70)

            data = result['data']

            print(f"\n분석 ID: {data['analysis_id']}")
            print(f"분석 시각: {data['analyzed_at']}")
            print(f"이미지 크기: {data['image_size']['width']}x{data['image_size']['height']}")
            print(f"총 메시지 수: {data['total_messages']}")

            print("\n" + "-"*70)
            print("추출된 메시지:")
            print("-"*70)

            for msg in data['messages']:
                bubble_icon = "◀" if msg['speaker'] == 'interlocutor' else "▶"
                speaker_label = "대화상대" if msg['speaker'] == 'interlocutor' else "사용자"
                print(f"\n[메시지 #{msg['id']}] {bubble_icon} {speaker_label} ({msg['speaker']})")
                print(f"  텍스트: {msg['text']}")
                print(f"  신뢰도: {msg['confidence']:.3f}")
                print(f"  위치: x={msg['position']['x']:.1f}, y={msg['position']['y']:.1f}, "
                      f"w={msg['position']['width']:.1f}, h={msg['position']['height']:.1f}")

            print("\n" + "="*70)

        # JSON 저장
        output_file = "test_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n결과가 저장되었습니다: {output_file}")

    except urllib.error.HTTPError as e:
        print(f"\n✗ HTTP 에러 발생 (Status: {e.code})")
        print(f"응답: {e.read().decode('utf-8')}")

    except urllib.error.URLError:
        print(f"\n✗ API 서버에 연결할 수 없습니다.")
        print(f"서버가 실행 중인지 확인하세요: {api_url}")
        print("서버 실행 명령: python3 main.py")

    except Exception as e:
        print(f"\n✗ 에러 발생: {str(e)}")


def check_server_health(api_url: str = "http://localhost:8000"):
    """서버 헬스 체크"""
    try:
        req = urllib.request.Request(f"{api_url}/health", method='GET')
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status == 200:
                print(f"✓ 서버가 정상적으로 실행 중입니다: {api_url}")
                return True
            else:
                print(f"✗ 서버 응답 이상: {response.status}")
                return False
    except:
        print(f"✗ 서버에 연결할 수 없습니다: {api_url}")
        print("서버를 먼저 실행하세요: python3 main.py")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python test_client.py <image_path> [api_url]")
        print("예시: python test_client.py chat_screenshot.jpg")
        print("예시: python test_client.py chat.png http://localhost:8000")
        sys.exit(1)

    image_path = sys.argv[1]
    api_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"

    # 서버 헬스 체크
    print("서버 상태 확인 중...")
    if check_server_health(api_url):
        print()
        test_chat_ocr(image_path, api_url)
