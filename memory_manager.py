import json
import sqlite3
from datetime import datetime

DB_PATH = "humaeli_memory.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    # 1. User Profiles Table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            name TEXT,
            age_group TEXT,
            preferred_language TEXT,
            interests TEXT,
            communication_style TEXT,
            past_concerns TEXT,
            conversation_summary TEXT,
            memory_enabled INTEGER DEFAULT 1
        )
    """)

    # 2. Short-Term Session Messages Table (Strict User + Session Isolation)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            created_at TEXT
        )
    """)

    # 3. Long-Term Cross-Session Memories Table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS long_term_memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            fact TEXT,
            embedding TEXT,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()
    seed_dummy_profiles()


def seed_dummy_profiles():
    dummy_users = [
        (
            "user_101",
            "Rohan Verma",
            "18-24 (College Student)",
            "Hinglish",
            "Coding, Cricket, Fitness",
            "Warm, concise bullet points in Hinglish",
            "Exam stress and late-night sleep schedule",
            "No conversation summary yet.",
            1,
        ),
        (
            "user_102",
            "Priya Sharma",
            "25-34 (Working Professional)",
            "English",
            "Reading, Yoga, Journaling",
            "Calm, structured, empathetic paragraphs in English",
            "Work burnout and presentation anxiety",
            "No conversation summary yet.",
            1,
        ),
        (
            "user_103",
            "Kabir Joshi",
            "35-44 (Manager)",
            "English",
            "Running, Podcasts, Mindfulness",
            "Direct, practical, action-oriented short tips",
            "Trouble switching off work thoughts before sleep",
            "No conversation summary yet.",
            1,
        ),
    ]

    conn = get_connection()
    cur = conn.cursor()
    for u in dummy_users:
        cur.execute("""
            INSERT OR IGNORE INTO user_profiles
            (user_id, name, age_group, preferred_language, interests, communication_style, past_concerns, conversation_summary, memory_enabled)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, u)
    conn.commit()
    conn.close()


# --- Profile Functions ---
def get_all_users():
    conn = get_connection()
    rows = conn.execute("SELECT user_id, name FROM user_profiles ORDER BY user_id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_profile(user_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_profile(user_id, age_group, preferred_language, interests, communication_style, past_concerns, memory_enabled):
    conn = get_connection()
    conn.execute("""
        UPDATE user_profiles
        SET age_group = ?, preferred_language = ?, interests = ?,
            communication_style = ?, past_concerns = ?, memory_enabled = ?
        WHERE user_id = ?
    """, (age_group, preferred_language, interests, communication_style, past_concerns, int(memory_enabled), user_id))
    conn.commit()
    conn.close()


def update_summary(user_id, new_summary):
    conn = get_connection()
    conn.execute("UPDATE user_profiles SET conversation_summary = ? WHERE user_id = ?", (new_summary, user_id))
    conn.commit()
    conn.close()
def create_user(user_id, name, age_group, preferred_language, interests, communication_style, past_concerns, memory_enabled=1):
    conn = get_connection()
    cur = conn.cursor()
    existing = cur.execute("SELECT user_id FROM user_profiles WHERE user_id = ?", (user_id,)).fetchone()
    if existing:
        conn.close()
        return False
    cur.execute("""
        INSERT INTO user_profiles
        (user_id, name, age_group, preferred_language, interests, communication_style, past_concerns, conversation_summary, memory_enabled)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id.strip(),
        name.strip(),
        age_group.strip(),
        preferred_language,
        interests.strip(),
        communication_style.strip(),
        past_concerns.strip(),
        "No conversation summary yet.",
        int(memory_enabled),
    ))
    conn.commit()
    conn.close()
    return True

# --- Short-Term Session History Functions ---
def get_user_sessions(user_id):
    conn = get_connection()
    rows = conn.execute(
        "SELECT DISTINCT session_id FROM chat_messages WHERE user_id = ? ORDER BY session_id",
        (user_id,),
    ).fetchall()
    conn.close()
    sessions = [r["session_id"] for r in rows]
    return sessions if sessions else ["session_1"]


def get_session_messages(user_id, session_id):
    conn = get_connection()
    rows = conn.execute(
        "SELECT role, content FROM chat_messages WHERE user_id = ? AND session_id = ? ORDER BY id ASC",
        (user_id, session_id),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_message(user_id, session_id, role, content):
    conn = get_connection()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "INSERT INTO chat_messages (user_id, session_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
        (user_id, session_id, role, content, now),
    )
    conn.commit()
    conn.close()


def clear_session_history(user_id, session_id):
    conn = get_connection()
    conn.execute("DELETE FROM chat_messages WHERE user_id = ? AND session_id = ?", (user_id, session_id))
    conn.commit()
    conn.close()


# --- Long-Term Memory Functions (Cross-Session + Relevance Scoring + Privacy) ---
def save_long_term_memory(user_id, fact_text, embedding_model):
    profile = get_profile(user_id)
    if not profile or profile["memory_enabled"] == 0:
        return  # Memory disabled hai toh save mat karo

    conn = get_connection()
    existing = conn.execute("SELECT fact FROM long_term_memories WHERE user_id = ?", (user_id,)).fetchall()
    for row in existing:
        if row["fact"].lower().strip() == fact_text.lower().strip():
            conn.close()
            return  # Duplicate fact skip karo

    vec = embedding_model.embed_query(fact_text)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    conn.execute(
        "INSERT INTO long_term_memories (user_id, fact, embedding, created_at) VALUES (?, ?, ?, ?)",
        (user_id, fact_text, json.dumps(vec), now),
    )
    conn.commit()
    conn.close()


def search_long_term_memories(user_id, query, embedding_model, min_score=0.35, top_k=3):
    profile = get_profile(user_id)
    if not profile or profile["memory_enabled"] == 0:
        return []  # Memory OFF hone par kuch retrieve nahi hoga

    conn = get_connection()
    rows = conn.execute(
        "SELECT id, fact, embedding, created_at FROM long_term_memories WHERE user_id = ?",
        (user_id,),
    ).fetchall()
    conn.close()

    if not rows:
        return []

    q_vec = embedding_model.embed_query(query)
    scored = []

    for r in rows:
        m_vec = json.loads(r["embedding"])
        dot = 0.0
        q_norm = 0.0
        m_norm = 0.0
        for i in range(len(q_vec)):
            dot += q_vec[i] * m_vec[i]
            q_norm += q_vec[i] * q_vec[i]
            m_norm += m_vec[i] * m_vec[i]

        sim = dot / ((q_norm ** 0.5) * (m_norm ** 0.5) + 1e-9)
        if sim >= min_score:
            scored.append((sim, r["fact"]))

    scored.sort(reverse=True)
    results = []
    for sim, fact in scored[:top_k]:
        results.append(f"{fact} (relevance: {round(sim, 2)})")
    return results


def get_all_memories(user_id):
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, fact, created_at FROM long_term_memories WHERE user_id = ? ORDER BY id DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_memory_by_id(user_id, memory_id):
    conn = get_connection()
    # user_id check zaroori hai taaki ek user doosre ki memory delete na kar sake
    conn.execute("DELETE FROM long_term_memories WHERE id = ? AND user_id = ?", (memory_id, user_id))
    conn.commit()
    conn.close()