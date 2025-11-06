"""
SQLite Database Models and Setup
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import json


class Database:
    """SQLite 데이터베이스 매니저"""

    def __init__(self, db_path: str = "chat_sessions.db"):
        """
        Args:
            db_path: 데이터베이스 파일 경로
        """
        self.db_path = db_path
        self.init_database()

    @contextmanager
    def get_connection(self):
        """데이터베이스 연결 컨텍스트 매니저"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 딕셔너리처럼 접근 가능
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def init_database(self):
        """데이터베이스 초기화 및 테이블 생성"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 세션 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'processing',
                    total_screenshots INTEGER DEFAULT 0,
                    total_messages INTEGER DEFAULT 0,
                    relationship TEXT,
                    relationship_info TEXT
                )
            """)

            # 스크린샷 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS screenshots (
                    screenshot_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    upload_order INTEGER NOT NULL,
                    uploaded_at TEXT NOT NULL,
                    processed INTEGER DEFAULT 0,
                    image_width INTEGER,
                    image_height INTEGER,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
            """)

            # 메시지 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    screenshot_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    speaker TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    position_x REAL NOT NULL,
                    position_y REAL NOT NULL,
                    position_width REAL NOT NULL,
                    position_height REAL NOT NULL,
                    group_id INTEGER,
                    sequence_order INTEGER, -- 메시지 순서
                    score REAL,
                    emotional_tone TEXT,
                    impact_score REAL,
                    review_comment TEXT,
                    suggested_alternative TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
                    FOREIGN KEY (screenshot_id) REFERENCES screenshots(screenshot_id) ON DELETE CASCADE
                )
            """)

            # 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_position_y
                ON messages(position_y)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_sequence_order
                ON messages(session_id, sequence_order)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_screenshots_session
                ON screenshots(session_id, upload_order)
            """)

            conn.commit()
            print(f"Database initialized: {self.db_path}")

    # ========== Session Operations ==========

    def create_session(self, session_id: str) -> Dict[str, Any]:
        """새 세션 생성"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()

            cursor.execute("""
                INSERT INTO sessions (session_id, created_at, updated_at, status)
                VALUES (?, ?, ?, ?)
            """, (session_id, now, now, 'processing'))

            return {
                'session_id': session_id,
                'created_at': now,
                'updated_at': now,
                'status': 'processing',
                'total_screenshots': 0,
                'total_messages': 0
            }

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """세션 정보 조회"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM sessions WHERE session_id = ?
            """, (session_id,))

            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def update_session_status(self, session_id: str, status: str):
        """세션 상태 업데이트"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE sessions
                SET status = ?, updated_at = ?
                WHERE session_id = ?
            """, (status, datetime.now().isoformat(), session_id))

    def update_session_relationship(self, session_id: str, relationship: str, relationship_info: str):
        """세션의 relationship 정보 업데이트"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE sessions
                SET relationship = ?, relationship_info = ?, updated_at = ?
                WHERE session_id = ?
            """, (relationship, relationship_info, datetime.now().isoformat(), session_id))

    def update_session_counts(self, session_id: str):
        """세션의 스크린샷/메시지 개수 업데이트"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 스크린샷 개수
            cursor.execute("""
                SELECT COUNT(*) as cnt FROM screenshots WHERE session_id = ?
            """, (session_id,))
            screenshot_count = cursor.fetchone()['cnt']

            # 메시지 개수
            cursor.execute("""
                SELECT COUNT(*) as cnt FROM messages WHERE session_id = ?
            """, (session_id,))
            message_count = cursor.fetchone()['cnt']

            cursor.execute("""
                UPDATE sessions
                SET total_screenshots = ?, total_messages = ?, updated_at = ?
                WHERE session_id = ?
            """, (screenshot_count, message_count, datetime.now().isoformat(), session_id))

    # ========== Screenshot Operations ==========

    def add_screenshot(self, screenshot_id: str, session_id: str, file_path: str,
                      upload_order: int, image_width: int, image_height: int) -> Dict[str, Any]:
        """스크린샷 추가"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()

            cursor.execute("""
                INSERT INTO screenshots
                (screenshot_id, session_id, file_path, upload_order, uploaded_at,
                 image_width, image_height, processed)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0)
            """, (screenshot_id, session_id, file_path, upload_order, now,
                  image_width, image_height))

            return {
                'screenshot_id': screenshot_id,
                'session_id': session_id,
                'file_path': file_path,
                'upload_order': upload_order,
                'uploaded_at': now,
                'processed': False
            }

    def get_screenshots(self, session_id: str) -> List[Dict[str, Any]]:
        """세션의 모든 스크린샷 조회 (순서대로)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM screenshots
                WHERE session_id = ?
                ORDER BY upload_order
            """, (session_id,))

            return [dict(row) for row in cursor.fetchall()]

    def mark_screenshot_processed(self, screenshot_id: str):
        """스크린샷 처리 완료 표시"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE screenshots SET processed = 1 WHERE screenshot_id = ?
            """, (screenshot_id,))

    # ========== Message Operations ==========

    def add_message(self, message_id: str, session_id: str, screenshot_id: str,
                   text: str, speaker: str, confidence: float,
                   position_x: float, position_y: float,
                   position_width: float, position_height: float,
                   group_id: Optional[int] = None,
                   sequence_order: Optional[int] = None) -> Dict[str, Any]:
        """메시지 추가"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()

            cursor.execute("""
                INSERT INTO messages
                (message_id, session_id, screenshot_id, text, speaker, confidence,
                 position_x, position_y, position_width, position_height,
                 group_id, sequence_order, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (message_id, session_id, screenshot_id, text, speaker, confidence,
                  position_x, position_y, position_width, position_height,
                  group_id, sequence_order, now))

            return {
                'message_id': message_id,
                'session_id': session_id,
                'screenshot_id': screenshot_id,
                'text': text,
                'speaker': speaker,
                'confidence': confidence,
                'position': {
                    'x': position_x,
                    'y': position_y,
                    'width': position_width,
                    'height': position_height
                },
                'group_id': group_id,
                'sequence_order': sequence_order,
                'created_at': now
            }

    def get_messages(self, session_id: str, order_by: str = 'sequence_order') -> List[Dict[str, Any]]:
        """세션의 모든 메시지 조회"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 정렬 기준이 유효한지 확인 (SQL 인젝션 방지)
            valid_order_by = ['sequence_order', 'position_y', 'created_at']
            if order_by not in valid_order_by:
                order_by = 'sequence_order'

            query = f"""
                SELECT * FROM messages
                WHERE session_id = ?
                ORDER BY {order_by}
            """
            cursor.execute(query, (session_id,))

            messages = []
            for row in cursor.fetchall():
                msg = dict(row)
                # position을 딕셔너리로 변환
                msg['position'] = {
                    'x': msg.pop('position_x'),
                    'y': msg.pop('position_y'),
                    'width': msg.pop('position_width'),
                    'height': msg.pop('position_height')
                }
                messages.append(msg)

            return messages

    def get_messages_by_screenshot(self, screenshot_id: str) -> List[Dict[str, Any]]:
        """특정 스크린샷의 메시지 조회"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM messages
                WHERE screenshot_id = ?
                ORDER BY position_y
            """, (screenshot_id,))

            messages = []
            for row in cursor.fetchall():
                msg = dict(row)
                msg['position'] = {
                    'x': msg.pop('position_x'),
                    'y': msg.pop('position_y'),
                    'width': msg.pop('position_width'),
                    'height': msg.pop('position_height')
                }
                messages.append(msg)

            return messages

    def update_message_score(self, message_id: str, score: float, ai_message: str):
        """메시지에 score와 ai_message 업데이트"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE messages
                SET score = ?, ai_message = ?
                WHERE message_id = ?
            """, (score, ai_message, message_id))

    def bulk_update_scores_by_group(self, session_id: str, updates: List[Dict[str, Any]]):
        """group_id 기반으로 메시지 정보 일괄 업데이트"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for update in updates:
                cursor.execute("""
                    UPDATE messages
                    SET score = ?,
                        emotional_tone = ?,
                        impact_score = ?,
                        review_comment = ?,
                        suggested_alternative = ?
                    WHERE session_id = ? AND group_id = ?
                """, (
                    update.get('score'),
                    update.get('emotional_tone'),
                    update.get('impact_score'),
                    update.get('review_comment'),
                    update.get('suggested_alternative'),
                    session_id,
                    update.get('group_id')
                ))

    def message_exists(self, session_id: str, text: str, position_y: float,
                      tolerance: float = 50.0) -> Optional[Dict[str, Any]]:
        """비슷한 위치에 같은 텍스트가 있는지 확인 (중복 체크)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM messages
                WHERE session_id = ?
                  AND text = ?
                  AND ABS(position_y - ?) < ?
                LIMIT 1
            """, (session_id, text, position_y, tolerance))

            row = cursor.fetchone()
            if row:
                msg = dict(row)
                msg['position'] = {
                    'x': msg.pop('position_x'),
                    'y': msg.pop('position_y'),
                    'width': msg.pop('position_width'),
                    'height': msg.pop('position_height')
                }
                return msg
            return None

    # ========== Search Operations ==========

    def search_messages_by_text(self, session_id: str, query: str) -> List[Dict[str, Any]]:
        """텍스트로 메시지 검색"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM messages
                WHERE session_id = ? AND text LIKE ?
                ORDER BY position_y
            """, (session_id, f'%{query}%'))

            messages = []
            for row in cursor.fetchall():
                msg = dict(row)
                msg['position'] = {
                    'x': msg.pop('position_x'),
                    'y': msg.pop('position_y'),
                    'width': msg.pop('position_width'),
                    'height': msg.pop('position_height')
                }
                messages.append(msg)

            return messages
