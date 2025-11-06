"""
Screenshot Merge Logic - 겹치는 메시지를 찾아서 이어붙이기
"""

from typing import List, Dict, Any, Tuple, Optional
from difflib import SequenceMatcher


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    두 텍스트의 유사도 계산 (0.0 ~ 1.0)

    Args:
        text1: 첫 번째 텍스트
        text2: 두 번째 텍스트

    Returns:
        유사도 (0.0 ~ 1.0)
    """
    return SequenceMatcher(None, text1, text2).ratio()


def find_overlap_messages(messages1: List[Dict[str, Any]],
                          messages2: List[Dict[str, Any]],
                          min_overlap: int = 2,
                          similarity_threshold: float = 0.9) -> Tuple[int, int, int]:
    """
    두 메시지 리스트에서 겹치는 부분을 찾기

    Args:
        messages1: 첫 번째 스크린샷의 메시지 리스트 (시간순 정렬)
        messages2: 두 번째 스크린샷의 메시지 리스트 (시간순 정렬)
        min_overlap: 최소 겹쳐야 하는 메시지 개수
        similarity_threshold: 텍스트 유사도 임계값

    Returns:
        (overlap_start_idx1, overlap_start_idx2, overlap_length)
        - overlap_start_idx1: messages1에서 겹치기 시작하는 인덱스
        - overlap_start_idx2: messages2에서 겹치기 시작하는 인덱스
        - overlap_length: 겹치는 메시지 개수
    """
    best_overlap = (0, 0, 0)  # (idx1, idx2, length)

    # messages1의 뒷부분과 messages2의 앞부분이 겹칠 가능성이 높음
    # messages1의 뒷부분부터 탐색
    search_start1 = max(0, len(messages1) - min(50, len(messages1)))

    for i in range(search_start1, len(messages1)):
        # messages2의 앞부분 탐색
        search_end2 = min(50, len(messages2))

        for j in range(search_end2):
            # i번째 messages1과 j번째 messages2부터 비교
            overlap_length = 0
            idx1 = i
            idx2 = j

            while idx1 < len(messages1) and idx2 < len(messages2):
                msg1 = messages1[idx1]
                msg2 = messages2[idx2]

                # 텍스트 유사도 체크
                similarity = calculate_text_similarity(msg1['text'], msg2['text'])

                if similarity >= similarity_threshold:
                    # speaker도 같아야 함
                    if msg1['speaker'] == msg2['speaker']:
                        overlap_length += 1
                        idx1 += 1
                        idx2 += 1
                    else:
                        break
                else:
                    break

            # 최소 겹침 개수를 만족하고, 이전보다 더 긴 겹침을 찾았으면 업데이트
            if overlap_length >= min_overlap and overlap_length > best_overlap[2]:
                best_overlap = (i, j, overlap_length)

    return best_overlap


def merge_two_screenshots(messages1: List[Dict[str, Any]],
                         messages2: List[Dict[str, Any]],
                         min_overlap: int = 2) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    두 스크린샷의 메시지를 병합

    Args:
        messages1: 첫 번째 스크린샷의 메시지 리스트
        messages2: 두 번째 스크린샷의 메시지 리스트
        min_overlap: 최소 겹쳐야 하는 메시지 개수

    Returns:
        (merged_messages, merge_info)
        - merged_messages: 병합된 메시지 리스트
        - merge_info: 병합 정보 (overlap_found, overlap_length 등)
    """
    if not messages1:
        return messages2, {'overlap_found': False, 'overlap_length': 0}

    if not messages2:
        return messages1, {'overlap_found': False, 'overlap_length': 0}

    # 겹치는 부분 찾기
    overlap_idx1, overlap_idx2, overlap_length = find_overlap_messages(
        messages1, messages2, min_overlap=min_overlap
    )

    merge_info = {
        'overlap_found': overlap_length >= min_overlap,
        'overlap_length': overlap_length,
        'overlap_start_idx1': overlap_idx1,
        'overlap_start_idx2': overlap_idx2
    }

    if overlap_length >= min_overlap:
        # 겹침이 발견됨
        # messages1의 overlap 이전 부분 + messages1의 overlap 부분 + messages2의 overlap 이후 부분
        merged = (
            messages1[:overlap_idx1] +
            messages2[overlap_idx2:]
        )

        print(f"  ✓ 겹침 발견: {overlap_length}개 메시지")
        print(f"    - Screenshot 1 (기존): index {overlap_idx1} 부터")
        print(f"    - Screenshot 2 (신규): index {overlap_idx2} 부터")
        print(f"    - 병합 전략: 기존 메시지에서 {overlap_idx1} 이전까지 + 신규 메시지 {overlap_idx2} 부터 전체")
        print(f"    - 중복 메시지 상세:")
        for i in range(overlap_length):
            msg1 = messages1[overlap_idx1 + i]
            msg2 = messages2[overlap_idx2 + i]
            sim = calculate_text_similarity(msg1['text'], msg2['text'])
            print(f"      - '{msg1['text']}' (sim: {sim:.2f}) '{msg2['text']}'")


        return merged, merge_info
    else:
        # 겹침이 없으면 그냥 이어붙이기
        print(f"  ⚠ 겹침 없음 - 순서대로 이어붙임")
        merged = messages1 + messages2
        return merged, merge_info


def merge_multiple_screenshots(screenshot_messages: List[List[Dict[str, Any]]],
                               min_overlap: int = 2) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    여러 스크린샷의 메시지를 순차적으로 병합

    Args:
        screenshot_messages: 각 스크린샷의 메시지 리스트들 (순서대로)
        min_overlap: 최소 겹쳐야 하는 메시지 개수

    Returns:
        (merged_messages, merge_history)
        - merged_messages: 최종 병합된 메시지 리스트
        - merge_history: 각 병합 단계의 정보
    """
    if not screenshot_messages:
        return [], []

    if len(screenshot_messages) == 1:
        return screenshot_messages[0], []

    print(f"\n=== 스크린샷 병합 시작 ({len(screenshot_messages)}개) ===")

    merged = screenshot_messages[0]
    merge_history = []

    for i in range(1, len(screenshot_messages)):
        print(f"\n[{i}/{len(screenshot_messages) - 1}] 병합 중...")
        print(f"  현재 누적: {len(merged)}개 메시지")
        print(f"  추가 대상: {len(screenshot_messages[i])}개 메시지")

        merged, merge_info = merge_two_screenshots(
            merged,
            screenshot_messages[i],
            min_overlap=min_overlap
        )

        merge_info['step'] = i
        merge_info['before_count'] = len(merged) - len(screenshot_messages[i]) + merge_info.get('overlap_length', 0)
        merge_info['after_count'] = len(merged)
        merge_history.append(merge_info)

        print(f"  병합 후: {len(merged)}개 메시지")

    print(f"\n=== 병합 완료: 총 {len(merged)}개 메시지 ===")

    return merged, merge_history


def normalize_message_positions(messages: List[Dict[str, Any]],
                                screenshot_heights: List[int],
                                merge_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    병합된 메시지의 Y 좌표를 전체 대화 내역에서의 절대 위치로 정규화

    Args:
        messages: 병합된 메시지 리스트
        screenshot_heights: 각 스크린샷의 높이 리스트
        merge_history: 병합 히스토리

    Returns:
        정규화된 메시지 리스트
    """
    # 간단하게 각 메시지에 순서 번호만 부여
    # (복잡한 정규화는 실제 사용 패턴에 따라 조정 가능)
    for idx, msg in enumerate(messages):
        msg['sequence_number'] = idx + 1

    return messages


def deduplicate_messages(messages: List[Dict[str, Any]],
                        similarity_threshold: float = 0.95) -> List[Dict[str, Any]]:
    """
    중복 메시지 제거 (혹시 모를 중복을 위한 안전장치)

    Args:
        messages: 메시지 리스트
        similarity_threshold: 중복 판단 유사도 임계값

    Returns:
        중복 제거된 메시지 리스트
    """
    if not messages:
        return []

    unique_messages = [messages[0]]
    duplicates_removed = 0

    for i in range(1, len(messages)):
        current_msg = messages[i]
        is_duplicate = False

        # 바로 이전 메시지와만 비교 (인접 중복만 제거)
        prev_msg = unique_messages[-1]

        similarity = calculate_text_similarity(current_msg['text'], prev_msg['text'])

        if similarity >= similarity_threshold and current_msg['speaker'] == prev_msg['speaker']:
            is_duplicate = True
            duplicates_removed += 1

        if not is_duplicate:
            unique_messages.append(current_msg)

    if duplicates_removed > 0:
        print(f"  중복 제거: {duplicates_removed}개 메시지")

    return unique_messages


def assign_global_group_ids(messages: List[Dict[str, Any]],
                           gap_threshold: int = 100) -> List[Dict[str, Any]]:
    """
    전체 병합된 메시지에 대해 group_id 재할당.
    병합 후에는 position 정보가 정확하지 않으므로, speaker 변경을 기준으로 그룹을 나눔.

    Args:
        messages: 메시지 리스트
        gap_threshold: (사용 안 함)

    Returns:
        group_id가 업데이트된 메시지 리스트
    """
    if not messages:
        return []

    print("  - 병합 후 메시지 그룹 ID 재할당 중...")
    current_group_id = 1
    messages[0]['group_id'] = current_group_id

    for i in range(1, len(messages)):
        current_msg = messages[i]
        prev_msg = messages[i - 1]

        # 스피커가 동일하면 같은 그룹으로 유지
        if current_msg['speaker'] == prev_msg['speaker']:
            current_msg['group_id'] = current_group_id
        else:
            # 스피커가 바뀌면 새 그룹 시작
            current_group_id += 1
            current_msg['group_id'] = current_group_id
    
    print(f"  - 총 {current_group_id}개 그룹 생성됨.")
    return messages
