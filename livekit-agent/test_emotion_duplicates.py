#!/usr/bin/env python3
"""
Test script to verify duplicate emotion detection functionality
"""
import asyncio
import json
from unittest.mock import Mock, MagicMock
from meeting_session import MeetingSession
from model_managers.mom_manager import MoMManager
from model_managers.transcription_manager import TranscriptionManager


class MockDeepEmotionRecognizer:
    """Mock emotion recognizer for testing"""
    def __init__(self):
        self.call_count = 0
        self.emotion_sequence = ["happy", "happy", "sad", "sad", "happy", "neutral", "neutral"]
    
    def analyze(self, img, participant_sid):
        """Return predefined emotion sequence for testing"""
        emotion = self.emotion_sequence[self.call_count % len(self.emotion_sequence)]
        self.call_count += 1
        return emotion


class TestMeetingSession:
    """Test class for meeting session emotion detection"""
    
    def __init__(self):
        self.sent_emotions = []
        self.setup_session()
    
    def setup_session(self):
        """Setup a mock meeting session for testing"""
        mom_manager = Mock()
        transcription_manager = Mock()
        
        self.session = MeetingSession("test_room", mom_manager, transcription_manager, None)
        
        # Replace emotion recognizer with mock
        self.session.emotion_recognizer = MockDeepEmotionRecognizer()
        
        # Mock the send_emotion_sync method to capture sent emotions
        original_send = self.session.send_emotion_sync
        def mock_send_emotion_sync(identity, emotion):
            self.sent_emotions.append({"identity": identity, "emotion": emotion})
            print(f"📤 MOCK SENT: {identity} -> {emotion}")
        
        self.session.send_emotion_sync = mock_send_emotion_sync
        
        # Setup participant mapping
        self.session.participant_identity_map["participant_1"] = "Alice"
        self.session.participant_identity_map["participant_2"] = "Bob"
    
    def simulate_emotion_detection(self, participant_sid, num_frames=7):
        """Simulate emotion detection for multiple frames"""
        print(f"\n🎭 Testing emotion detection for {participant_sid}")
        print("=" * 50)
        
        for i in range(num_frames):
            # Mock frame data (not actually used by our mock)
            frame_data = b"mock_frame_data"
            width, height = 640, 480
            timestamp = i * 0.1  # 100ms intervals
            
            print(f"\n📸 Frame {i+1}:")
            emotion = self.session.analyze_emotion_sync(frame_data, width, height, participant_sid, timestamp)
            print(f"   Detected: {emotion}")
    
    def analyze_results(self):
        """Analyze the test results"""
        print(f"\n📊 ANALYSIS")
        print("=" * 50)
        print(f"Total emotions sent: {len(self.sent_emotions)}")
        print(f"Emotion sequence was: {self.session.emotion_recognizer.emotion_sequence}")
        
        print(f"\nSent emotions:")
        for i, sent in enumerate(self.sent_emotions):
            print(f"  {i+1}. {sent['identity']}: {sent['emotion']}")
        
        # Expected: happy, sad, happy, neutral (4 unique emotions, duplicates filtered)
        expected_emotions = ["happy", "sad", "happy", "neutral"]
        actual_emotions = [sent["emotion"] for sent in self.sent_emotions if sent["identity"] == "Alice"]
        
        print(f"\nExpected emotions for Alice: {expected_emotions}")
        print(f"Actual emotions for Alice: {actual_emotions}")
        
        if actual_emotions == expected_emotions:
            print("✅ TEST PASSED: Duplicate emotions correctly filtered!")
        else:
            print("❌ TEST FAILED: Unexpected emotion sequence!")
        
        return actual_emotions == expected_emotions


def main():
    """Run the duplicate emotion detection test"""
    print("🧪 Testing Duplicate Emotion Detection")
    print("=" * 60)
    
    test = TestMeetingSession()
    
    # Test with participant 1 (Alice)
    test.simulate_emotion_detection("participant_1", 7)
    
    # Test with participant 2 (Bob) - should have separate tracking
    print(f"\n🎭 Testing emotion detection for participant_2")
    print("=" * 50)
    test.simulate_emotion_detection("participant_2", 3)
    
    # Analyze results
    success = test.analyze_results()
    
    # Check participant-specific tracking
    alice_emotions = [sent for sent in test.sent_emotions if sent["identity"] == "Alice"]
    bob_emotions = [sent for sent in test.sent_emotions if sent["identity"] == "Bob"]
    
    print(f"\nAlice sent {len(alice_emotions)} emotions")
    print(f"Bob sent {len(bob_emotions)} emotions")
    
    if len(alice_emotions) > 0 and len(bob_emotions) > 0:
        print("✅ Per-participant tracking works correctly!")
    else:
        print("❌ Per-participant tracking failed!")
    
    print(f"\n🏁 Overall test result: {'PASSED' if success else 'FAILED'}")


if __name__ == "__main__":
    main()
