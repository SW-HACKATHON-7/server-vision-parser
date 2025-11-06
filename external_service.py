"""
External Service Integration - 외부 서버에서 score/message 받아오기
"""

import asyncio
import random
from typing import List, Dict, Any, Optional
import aiohttp
import json


class ExternalScoreService:
    """외부 서버 연동 서비스"""

    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Args:
            api_url: 외부 API URL (None이면 더미 데이터 사용)
            api_key: API 인증 키
        """
        self.api_url = api_url
        self.api_key = api_key
        self.use_dummy = api_url is None

    async def get_scores_for_messages(self, messages: List[Dict[str, Any]], relationship: str, relationship_info: str) -> List[Dict[str, Any]]:
        """
        모든 메시지들에 대해 score와 ai_message를 받아옴

        Args:
            messages: 전체 메시지 리스트 (user, interlocutor 모두 포함)
            relationship: 대화 상대와의 관계 (예: "FRIEND", "SUPERIOR")
            relationship_info: 관계에 대한 추가 정보 (예: "2년 지기", "신입사원")

        Returns:
            score 정보 리스트 [{'message_id': ..., 'score': ..., 'ai_message': ...}]
        """
        if not messages:
            print("  ℹ 메시지가 없어서 외부 API 호출 스킵")
            return []

        print(f"\n=== 외부 서버 연동 시작 ({len(messages)}개 메시지) ===")
        print(f"  대화 상대: {relationship} ({relationship_info})")

        if self.use_dummy:
            return await self._get_dummy_scores(messages)
        else:
            return await self._call_external_api(messages, relationship, relationship_info)

    async def _get_dummy_scores(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        더미 데이터 생성 (실제 외부 API 없을 때)

        Args:
            messages: 모든 메시지 리스트 (user, interlocutor 포함)

        Returns:
            group_id 기반 더미 score 정보 리스트
        """
        print("  [DUMMY MODE] 더미 score/message 생성 중...")

        # group_id별로 그룹화
        groups = {}
        for msg in messages:
            group_id = msg.get('group_id')
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(msg)

        results = []
        dummy_comments = [
            "공감능력이 부족해 보입니다.",
            "상대방의 감정을 잘 이해하고 있습니다.",
            "조금 더 부드러운 표현을 사용해보세요.",
            "긍정적인 대화 태도입니다.",
            "상대방을 배려하는 좋은 표현입니다.",
            "감정적인 표현보다는 논리적으로 접근해보세요.",
            "훌륭한 커뮤니케이션입니다.",
            "약간 방어적인 태도가 보입니다.",
        ]

        dummy_alternatives = [
            "'그 부분이 조금 의아했는데, 네 생각을 좀 더 듣고 싶어.'라고 말해보세요.",
            "'이해가 안 되는 부분이 있어서, 조금 더 설명해줄 수 있을까?'라고 물어보세요.",
            None,  # 대안 제시 없음
        ]

        tones = ["POSITIVE", "NEUTRAL", "NEGATIVE"]

        # 각 그룹별로 더미 데이터 생성
        for group_id in groups.keys():
            score = round(random.uniform(20, 95), 0)
            impact_score = round(random.uniform(-3, 3), 0)
            emotional_tone = random.choice(tones)
            review_comment = random.choice(dummy_comments)
            suggested_alternative = random.choice(dummy_alternatives)

            results.append({
                'group_id': group_id,
                'score': score,
                'emotional_tone': emotional_tone,
                'impact_score': impact_score,
                'review_comment': review_comment,
                'suggested_alternative': suggested_alternative
            })

        # 외부 API 호출 시뮬레이션 (약간의 지연)
        await asyncio.sleep(0.5)

        print(f"  ✓ 더미 score 생성 완료: {len(results)}개 그룹")

        return results

    async def _call_external_api(self, messages: List[Dict[str, Any]], relationship: str, relationship_info: str) -> List[Dict[str, Any]]:
        """
        실제 외부 API 호출

        Args:
            messages: 모든 메시지 리스트 (user, interlocutor 포함)
            relationship: 대화 상대와의 관계
            relationship_info: 관계에 대한 추가 정보

        Returns:
            API 응답 데이터
        """
        print(f"  외부 API 호출: {self.api_url}")

        # 요청 데이터 구성
        request_data = {
            'relationship': relationship,
            'relationship_info': relationship_info,
            'messages': [
                {
                    'message_id': msg.get('message_id'),
                    'text': msg.get('text'),
                    'speaker': msg.get('speaker'),
                    'confidence': msg.get('confidence'),
                    'group_id': msg.get('group_id')
                }
                for msg in messages
            ]
        }
        
        print("  - API 요청 데이터:")
        print(json.dumps(request_data, indent=2, ensure_ascii=False))

        headers = {}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    json=request_data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        print("  - API 응답 데이터:")
                        print(json.dumps(data, indent=2, ensure_ascii=False))

                        # response 배열이면 그대로 사용, 아니면 빈 배열
                        if isinstance(data, list):
                            results = data
                        else:
                            results = data.get('response', data.get('results', []))
                        print(f"  ✓ API 호출 성공: {len(results)}개 그룹 응답")
                        return results
                    else:
                        error_text = await response.text()
                        print(f"  ✗ API 호출 실패: HTTP {response.status}")
                        print(f"  - 응답 내용: {error_text}")
                        # 실패 시 더미 데이터로 fallback
                        return await self._get_dummy_scores(messages)

        except asyncio.TimeoutError:
            print("  ✗ API 호출 타임아웃 - 더미 데이터 사용")
            return await self._get_dummy_scores(messages)
        except Exception as e:
            print(f"  ✗ API 호출 에러: {str(e)} - 더미 데이터 사용")
            return await self._get_dummy_scores(messages)

    async def analyze_conversation_quality(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        전체 대화 품질 분석 (추가 기능)

        Args:
            messages: 전체 메시지 리스트

        Returns:
            대화 품질 분석 결과
        """
        user_messages = [msg for msg in messages if msg.get('speaker') == 'user']
        interlocutor_messages = [msg for msg in messages if msg.get('speaker') == 'interlocutor']

        # 더미 분석 결과
        analysis = {
            'total_messages': len(messages),
            'user_message_count': len(user_messages),
            'interlocutor_message_count': len(interlocutor_messages),
            'average_user_score': 0.0,
            'conversation_balance': len(user_messages) / len(messages) if messages else 0.0,
            'suggestions': []
        }

        # 실제 외부 API가 있다면 여기서 호출
        if not self.use_dummy:
            # TODO: 실제 API 호출
            pass

        return analysis


def get_external_service(api_url: Optional[str] = None,
                        api_key: Optional[str] = None) -> ExternalScoreService:
    """
    외부 서비스 인스턴스 생성

    Args:
        api_url: 외부 API URL (None이면 더미 모드)
        api_key: API 인증 키

    Returns:
        ExternalScoreService 인스턴스
    """
    return ExternalScoreService(api_url, api_key)
