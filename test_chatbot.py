# test_chatbot.py
import requests
import json
import time

def test_chatbot():
    # Wait a moment for server to fully start
    time.sleep(2)
    
    # Test health endpoint
    try:
        response = requests.get('http://127.0.0.1:5000/api/health', timeout=10)
        print("Health check:", response.json())
    except Exception as e:
        print(f"Health check failed: {e}")
        return
    
    # Test chat endpoint
    messages = [
        "I have a headache and fever",
        "My stomach hurts and I feel nauseous",
        "I have cough and cold with chest pain",
        "I need a dermatologist in Dhaka"
    ]
    
    for message in messages:
        try:
            response = requests.post(
                'http://127.0.0.1:5000/api/chat',
                json={'message': message},
                timeout=10
            )
            print(f"\n💬 Message: {message}")
            print("✅ Response:", response.json())
        except Exception as e:
            print(f"❌ Error with message '{message}': {e}")

if __name__ == "__main__":
    test_chatbot()