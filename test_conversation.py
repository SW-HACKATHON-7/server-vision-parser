"""
대화 프록시 API 테스트 클라이언트
"""

import requests
import json
from typing import Dict, Any
import time


class ConversationClient:
    """대화 프록시 API 클라이언트"""

    def __init__(self, base_url: str = "http://3.239.81.172"):
        """
        Args:
            base_url: API 서버 URL
        """
        self.base_url = base_url
        self.thread_id = None

    def start_conversation(self, relationship: str = "연인") -> Dict[str, Any]:
        """대화 시작"""
        print("\n" + "="*70)
        print("1. 대화 시작")
        print("="*70)

        print(f"관계: {relationship}")

        response = requests.post(
            f"{self.base_url}/start-conversation",
            json={"relationship": relationship}
        )

        # 에러 응답 상세 출력
        if response.status_code != 200:
            print(f"\n❌ HTTP {response.status_code} 에러")
            print(f"응답 내용:")
            try:
                error_data = response.json()
                print(json.dumps(error_data, indent=2, ensure_ascii=False))
            except:
                print(response.text)
            response.raise_for_status()

        data = response.json()
        self.thread_id = data['thread_id']

        print(f"\n✓ 대화 시작 완료")
        print(f"  Thread ID: {self.thread_id}")
        print(f"  AI: \"{data['message']}\"")

        return data

    def continue_conversation(self, message: str) -> Dict[str, Any]:
        """대화 이어가기"""
        if not self.thread_id:
            raise ValueError("대화를 먼저 시작하세요. start_conversation()을 호출하세요.")

        print(f"\n{'─'*70}")
        print(f"💬 User: \"{message}\"")

        response = requests.post(
            f"{self.base_url}/continue-conversation",
            json={
                "message": message,
                "thread_id": self.thread_id
            }
        )

        # 에러 응답 상세 출력
        if response.status_code != 200:
            print(f"\n❌ HTTP {response.status_code} 에러")
            print(f"응답 내용:")
            try:
                error_data = response.json()
                print(json.dumps(error_data, indent=2, ensure_ascii=False))
            except:
                print(response.text)
            response.raise_for_status()

        data = response.json()

        print(f"🤖 AI: \"{data['message']}\"")

        if data.get('response'):
            resp = data['response']
            print(f"\n📊 평가:")
            print(f"  감정 톤: {resp.get('emotional_tone')}")
            print(f"  적절성: {resp.get('appropriateness_rating')}/100")
            print(f"  영향: {resp.get('impact_score')}")
            print(f"  피드백: {resp.get('review_comment')}")
            if resp.get('suggested_alternative'):
                print(f"  💡 추천: {resp.get('suggested_alternative')}")

        return data


def test_simple_conversation():
    """간단한 대화 테스트"""
    print("="*70)
    print("    대화 프록시 API 테스트 - 간단한 시나리오")
    print("="*70)

    try:
        client = ConversationClient()

        # 1. 대화 시작
        client.start_conversation(relationship="연인")

        # 2. 대화 이어가기
        messages = [
            "요즘 좀 바빠",
            "미안해 좀 더 신경 쓸게",
        ]

        for msg in messages:
            time.sleep(0.5)  # 잠시 대기
            client.continue_conversation(msg)

        print("\n" + "="*70)
        print("✓ 테스트 완료!")
        print("="*70)

    except requests.exceptions.ConnectionError:
        print("\n❌ API 서버에 연결할 수 없습니다.")
        print("   서버가 실행 중인지 확인하세요.")

    except Exception as e:
        print(f"\n❌ 에러 발생: {str(e)}")
        import traceback
        traceback.print_exc()


def test_multiple_relationships():
    """여러 관계 유형 테스트"""
    print("="*70)
    print("    대화 프록시 API 테스트 - 다양한 관계")
    print("="*70)

    relationships = ["연인", "친구", "상사", "부모"]

    for rel in relationships:
        print(f"\n{'='*70}")
        print(f"관계: {rel}")
        print(f"{'='*70}")

        try:
            client = ConversationClient()
            result = client.start_conversation(relationship=rel)
            print(f"✓ {rel} 관계 대화 시작 성공")

            # 한 턴만 테스트
            client.continue_conversation("안녕하세요")

            time.sleep(1)  # API 부하 방지

        except Exception as e:
            print(f"✗ {rel} 관계 테스트 실패: {str(e)}")

    print("\n" + "="*70)
    print("✓ 모든 관계 테스트 완료!")
    print("="*70)


def test_long_conversation():
    """긴 대화 테스트"""
    print("="*70)
    print("    대화 프록시 API 테스트 - 긴 대화")
    print("="*70)

    try:
        client = ConversationClient()

        # 1. 대화 시작
        client.start_conversation(relationship="친구")

        # 2. 여러 턴 대화
        messages = [
            "오늘 기분이 별로야",
            "일이 너무 많아서 그래",
            "응 고마워 힘내볼게",
            "너는 요즘 어때?",
            "좋다니 다행이다"
        ]

        for i, msg in enumerate(messages, 1):
            print(f"\n[턴 {i}/{len(messages)}]")
            client.continue_conversation(msg)
            time.sleep(0.5)

        print("\n" + "="*70)
        print(f"✓ {len(messages)}턴 대화 테스트 완료!")
        print("="*70)

    except Exception as e:
        print(f"\n❌ 에러 발생: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """메인 테스트 함수"""
    import sys

    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        if test_type == "simple":
            test_simple_conversation()
        elif test_type == "multiple":
            test_multiple_relationships()
        elif test_type == "long":
            test_long_conversation()
        else:
            print(f"알 수 없는 테스트 타입: {test_type}")
            print("사용법: python test_conversation.py [simple|multiple|long]")
    else:
        # 기본: 간단한 테스트
        test_simple_conversation()


if __name__ == "__main__":
    main()
