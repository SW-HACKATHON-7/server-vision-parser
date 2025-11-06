"""
Test Client for Session-based Chat OCR API V2
"""

import requests
import json
from pathlib import Path
from typing import List, Dict, Any
import time


class ChatOCRClient:
    """Chat OCR API V2 클라이언트"""

    def __init__(self, base_url: str = "http://3.239.81.172"):
        """
        Args:
            base_url: API 서버 URL
        """
        self.base_url = base_url
        self.session_id = None

    def create_session(self) -> Dict[str, Any]:
        """새 세션 생성"""
        print("\n" + "="*70)
        print("1. 세션 생성")
        print("="*70)

        response = requests.post(f"{self.base_url}/sessions")
        response.raise_for_status()

        data = response.json()
        self.session_id = data['session_id']

        print(f"✓ 세션 생성 완료")
        print(f"  Session ID: {self.session_id}")
        print(f"  Created at: {data['created_at']}")

        return data

    def upload_screenshots(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """스크린샷 업로드"""
        print("\n" + "="*70)
        print("2. 스크린샷 업로드")
        print("="*70)

        if not self.session_id:
            raise ValueError("세션이 생성되지 않았습니다. create_session()을 먼저 호출하세요.")

        results = []

        for idx, image_path in enumerate(image_paths, 1):
            print(f"\n[{idx}/{len(image_paths)}] 업로드: {image_path}")

            if not Path(image_path).exists():
                print(f"  ⚠ 파일이 없습니다: {image_path}")
                continue

            with open(image_path, 'rb') as f:
                files = {'file': (Path(image_path).name, f, 'image/jpeg')}
                response = requests.post(
                    f"{self.base_url}/sessions/{self.session_id}/upload",
                    files=files
                )
                response.raise_for_status()

            data = response.json()
            results.append(data)

            print(f"  ✓ 업로드 완료")
            print(f"    Screenshot ID: {data['screenshot_id']}")
            print(f"    Upload Order: {data['upload_order']}")

        print(f"\n총 {len(results)}개 스크린샷 업로드 완료")
        return results

    def process_session(self, relationship: str = "FRIEND", relationship_info: str = "친한 친구") -> Dict[str, Any]:
        """세션 처리 (OCR + 병합 + 외부 API)"""
        print("\n" + "="*70)
        print("3. 세션 처리 (OCR + 병합 + 외부 API)")
        print("="*70)

        if not self.session_id:
            raise ValueError("세션이 생성되지 않았습니다.")

        print(f"대화 상대: {relationship} ({relationship_info})")
        print("처리 시작... (시간이 걸릴 수 있습니다)")
        start_time = time.time()

        response = requests.post(
            f"{self.base_url}/sessions/{self.session_id}/process",
            params={
                "relationship": relationship,
                "relationship_info": relationship_info
            }
        )
        response.raise_for_status()

        elapsed_time = time.time() - start_time
        data = response.json()

        print(f"\n✓ 세션 처리 완료 ({elapsed_time:.1f}초 소요)")
        print(f"  Status: {data['status']}")
        print(f"  Total Screenshots: {data['total_screenshots']}")
        print(f"  Total Messages: {data['total_messages']}")
        print(f"  External API Called: {data['external_api_called']}")

        # 병합 정보
        merge_info = data.get('merge_info', {})
        if merge_info.get('merge_history'):
            print(f"\n  병합 히스토리:")
            for history in merge_info['merge_history']:
                if history.get('overlap_found'):
                    print(f"    Step {history['step']}: {history['overlap_length']}개 메시지 겹침 발견")
                else:
                    print(f"    Step {history['step']}: 겹침 없음 (순서대로 이어붙임)")

        return data

    def get_messages(self) -> Dict[str, Any]:
        """메시지 조회"""
        print("\n" + "="*70)
        print("4. 메시지 조회")
        print("="*70)

        if not self.session_id:
            raise ValueError("세션이 생성되지 않았습니다.")

        response = requests.get(f"{self.base_url}/sessions/{self.session_id}/messages")
        response.raise_for_status()

        data = response.json()

        print(f"✓ 메시지 조회 완료")
        print(f"  Total Messages: {data['total_messages']}")
        print(f"  Total Screenshots: {data['total_screenshots']}")

        # 메시지 미리보기 (처음 5개 + 마지막 5개)
        messages = data['messages']

        if messages:
            print(f"\n  메시지 미리보기 (처음 5개):")
            for msg in messages[:5]:
                speaker_icon = "🟢" if msg['speaker'] == 'user' else "🔵"
                score_info = f" [score: {msg['score']:.1f}]" if msg.get('score') else ""
                print(f"    {speaker_icon} {msg['text'][:50]}{'...' if len(msg['text']) > 50 else ''}{score_info}")

            if len(messages) > 10:
                print(f"\n  ... ({len(messages) - 10}개 메시지 생략) ...\n")

            if len(messages) > 5:
                print(f"  메시지 미리보기 (마지막 5개):")
                for msg in messages[-5:]:
                    speaker_icon = "🟢" if msg['speaker'] == 'user' else "🔵"
                    score_info = f" [score: {msg['score']:.1f}]" if msg.get('score') else ""
                    print(f"    {speaker_icon} {msg['text'][:50]}{'...' if len(msg['text']) > 50 else ''}{score_info}")

            # user 메시지 통계
            user_messages = [msg for msg in messages if msg['speaker'] == 'user']
            if user_messages:
                scores = [msg['score'] for msg in user_messages if msg.get('score')]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    print(f"\n  User 메시지 평균 점수: {avg_score:.2f}")

        return data

    def search_by_screenshot(self, search_image_path: str) -> Dict[str, Any]:
        """스크린샷으로 검색"""
        print("\n" + "="*70)
        print("5. 스크린샷으로 검색")
        print("="*70)

        if not self.session_id:
            raise ValueError("세션이 생성되지 않았습니다.")

        if not Path(search_image_path).exists():
            raise FileNotFoundError(f"검색 이미지가 없습니다: {search_image_path}")

        print(f"검색 이미지: {search_image_path}")

        with open(search_image_path, 'rb') as f:
            files = {'file': (Path(search_image_path).name, f, 'image/jpeg')}
            response = requests.post(
                f"{self.base_url}/sessions/{self.session_id}/search",
                files=files
            )
            response.raise_for_status()

        data = response.json()

        print(f"✓ 검색 완료")
        print(f"  Matched: {data['matched']}")
        print(f"  Message: {data['message']}")

        if data.get('results'):
            print(f"\n  매칭된 메시지:")
            for msg in data['results'][:10]:  # 최대 10개만 출력
                speaker_icon = "🟢" if msg['speaker'] == 'user' else "🔵"
                print(f"    {speaker_icon} {msg['text'][:60]}{'...' if len(msg['text']) > 60 else ''}")

        return data

    def predict_next_message(self) -> Dict[str, Any]:
        """다음 대화 예측"""
        print("\n" + "="*70)
        print("7. 다음 대화 예측")
        print("="*70)

        if not self.session_id:
            raise ValueError("세션이 생성되지 않았습니다.")

        print("AI가 다음 대화를 예측하는 중...")

        response = requests.post(f"{self.base_url}/sessions/{self.session_id}/predict-next")
        response.raise_for_status()

        data = response.json()

        print(f"✓ 예측 완료")
        print(f"  Session ID: {data['session_id']}")
        print(f"  대화 상대: {data['relationship']} ({data['relationship_info']})")
        print(f"  분석된 메시지: {data['total_messages']}개")

        if data.get('suggestions'):
            print(f"\n  💡 추천 답변 (3가지):\n")
            for idx, suggestion in enumerate(data['suggestions'], 1):
                print(f"  [{idx}] {suggestion['style']}")
                print(f"      \"{suggestion['text']}\"")
                print(f"      → {suggestion['explanation']}")
                print(f"      예상 영향: {suggestion['expected_impact']}")
                print()

        return data

    def start_conversation(self, relationship: str = "연인") -> Dict[str, Any]:
        """대화 시작"""
        print("\n" + "="*70)
        print("대화 시작 (프록시)")
        print("="*70)

        print(f"관계: {relationship}")

        response = requests.post(
            f"{self.base_url}/start-conversation",
            json={"relationship": relationship}
        )
        response.raise_for_status()

        data = response.json()

        print(f"✓ 대화 시작 완료")
        print(f"  Thread ID: {data['thread_id']}")
        print(f"  AI 메시지: \"{data['message']}\"")

        return data

    def continue_conversation(self, message: str, thread_id: str) -> Dict[str, Any]:
        """대화 이어가기"""
        print("\n" + "="*70)
        print("대화 이어가기 (프록시)")
        print("="*70)

        print(f"Thread ID: {thread_id}")
        print(f"User 메시지: \"{message}\"")

        response = requests.post(
            f"{self.base_url}/continue-conversation",
            json={
                "message": message,
                "thread_id": thread_id
            }
        )
        response.raise_for_status()

        data = response.json()

        print(f"\n✓ 대화 이어가기 완료")
        print(f"  AI 메시지: \"{data['message']}\"")

        if data.get('response'):
            resp = data['response']
            print(f"\n  📊 평가 결과:")
            print(f"    - 감정 톤: {resp.get('emotional_tone')}")
            print(f"    - 적절성 평가: {resp.get('appropriateness_rating')}/100")
            print(f"    - 영향 점수: {resp.get('impact_score')}")
            print(f"    - 피드백: {resp.get('review_comment')}")
            if resp.get('suggested_alternative'):
                print(f"    - 추천 표현: {resp.get('suggested_alternative')}")

        return data

    def save_results_to_file(self, messages_data: Dict[str, Any], output_path: str = "session_result.json"):
        """결과를 파일로 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(messages_data, f, ensure_ascii=False, indent=2)

        print(f"\n✓ 결과 저장: {output_path}")


def main():
    """메인 테스트 함수"""
    print("="*70)
    print("    Chat OCR API V2 - 세션 기반 다중 스크린샷 테스트")
    print("="*70)

    # 이미지 파일 확인
    image_files = ["target.jpg", "target1.jpg"]
    existing_files = [f for f in image_files if Path(f).exists()]

    if not existing_files:
        print(f"\n⚠ 테스트 이미지 파일이 없습니다.")
        print(f"   다음 파일들을 준비해주세요: {', '.join(image_files)}")
        return

    print(f"\n📁 사용 가능한 이미지 파일: {', '.join(existing_files)}")

    try:
        # 클라이언트 생성
        client = ChatOCRClient()

        # 1. 세션 생성
        session_data = client.create_session()

        # 2. 스크린샷 업로드
        upload_results = client.upload_screenshots(existing_files)

        # 3. 세션 처리 (relationship, relationship_info 지정)
        process_result = client.process_session(
            relationship="FRIEND",
            relationship_info="2년 지기"
        )

        # 4. 메시지 조회
        messages_data = client.get_messages()

        # 5. 결과 저장
        output_file = f"session_{client.session_id}_result.json"
        client.save_results_to_file(messages_data, output_file)

        # 6. (선택) 첫 번째 이미지로 검색 테스트
        if existing_files:
            print("\n" + "="*70)
            print("6. 검색 기능 테스트 (첫 번째 이미지로)")
            print("="*70)
            search_result = client.search_by_screenshot(existing_files[0])

        # 7. 다음 대화 예측
        prediction_result = client.predict_next_message()

        print("\n" + "="*70)
        print("✓ 모든 테스트 완료!")
        print("="*70)
        print(f"\nSession ID: {client.session_id}")
        print(f"결과 파일: {output_file}")

    except requests.exceptions.ConnectionError:
        print("\n❌ API 서버에 연결할 수 없습니다.")
        print("   서버가 실행 중인지 확인하세요: python main.py")

    except Exception as e:
        print(f"\n❌ 에러 발생: {str(e)}")
        import traceback
        traceback.print_exc()


def test_conversation():
    """대화 프록시 기능 테스트"""
    print("="*70)
    print("    대화 프록시 API 테스트")
    print("="*70)

    try:
        client = ChatOCRClient()

        # 1. 대화 시작
        start_result = client.start_conversation(relationship="연인")
        thread_id = start_result['thread_id']

        # 2. 대화 이어가기 (여러 턴)
        messages = [
            "싫어",
            "요즘 바빠서 그래",
            "미안해 좀 더 신경 쓸게"
        ]

        for msg in messages:
            time.sleep(1)  # 잠시 대기
            client.continue_conversation(msg, thread_id)

        print("\n" + "="*70)
        print("✓ 대화 테스트 완료!")
        print("="*70)

    except requests.exceptions.ConnectionError:
        print("\n❌ API 서버에 연결할 수 없습니다.")
        print("   서버가 실행 중인지 확인하세요: python main.py")

    except Exception as e:
        print(f"\n❌ 에러 발생: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys

    # 인자로 'conversation' 전달 시 대화 테스트 실행
    if len(sys.argv) > 1 and sys.argv[1] == "conversation":
        test_conversation()
    else:
        main()
