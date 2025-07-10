import json
import os
from datetime import datetime

DB_PATH = "sessions/emotion/emotion_log.json"

def save_emotion_to_db(room_name, speaker, emotion, confidence=None):
    print(f"Saving emotion for {speaker} in room {room_name}: {emotion} (confidence: {confidence})")
    if not room_name or not speaker or not emotion:
        print("❌ Invalid data provided.")
        return

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    # Load existing data
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    now = datetime.utcnow()
    rounded_timestamp = now.replace(second=(now.second // 10) * 10, microsecond=0)

    # Filter out any entry from the same speaker in same 10-sec window
    data = [
        entry for entry in data
        if not (
            entry["room"] == room_name and
            entry["speaker"] == speaker and
            datetime.fromisoformat(entry["timestamp"]).replace(second=(datetime.fromisoformat(entry["timestamp"]).second // 10) * 10, microsecond=0)
            == rounded_timestamp
        )
    ]

    # Check if the last saved emotion for this speaker in this room is same
    last_emotion_entry = None
    for entry in reversed(data):
        if entry["room"] == room_name and entry["speaker"] == speaker:
            last_emotion_entry = entry
            break

    if last_emotion_entry:
        same_emotion = last_emotion_entry["emotion"] == emotion
        similar_confidence = (
            confidence is not None and 
            abs(round(confidence, 3) - round(last_emotion_entry.get("confidence", 0), 3)) < 0.01
        )

        if same_emotion and similar_confidence:
            print("⏩ Skipped saving duplicate emotion with similar confidence.")
            return

    # Add new entry
    new_entry = {
        "timestamp": now.isoformat(),
        "room": room_name,
        "speaker": speaker,
        "emotion": emotion
    }
    if confidence is not None:
        new_entry["confidence"] = round(confidence, 3)

    data.append(new_entry)

    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Emotion saved to {DB_PATH}")


def clear_room_emotions(room_name: str):
    if not os.path.exists(DB_PATH):
        return

    with open(DB_PATH, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = []

    new_data = [entry for entry in data if entry.get("room") != room_name]

    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2)

    print(f"🧹 Cleared all emotion logs for room: {room_name}")


def load_emotion_log():
    if not os.path.exists(DB_PATH):
        return []

    with open(DB_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print("⚠️ Error decoding JSON from emotion log.")
            return []
