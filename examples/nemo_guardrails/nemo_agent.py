
import os
import dotenv
from openai import OpenAI
import time
import uuid

dotenv.load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

# Open Sentinel Proxy URL
OSNTL_URL = os.getenv("OSNTL_URL", "http://localhost:4000/v1")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Session ID for tracking conversation state
SESSION_ID = f"nemo-session-{uuid.uuid4().hex[:8]}"

if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not found in environment")

# =============================================================================
# Agent Setup
# =============================================================================

def get_system_instruction():
    return """You are a helpful customer support agent for TechCo.
You help with refunds, subscriptions, and general questions.
"""

def create_client():
    """Create OpenAI client configured for Open Sentinel."""
    return OpenAI(
        base_url=OSNTL_URL,
        api_key=GOOGLE_API_KEY or "dummy" 
    )

# =============================================================================
# Example Conversation
# =============================================================================

def run_nemo_conversation():
    print("=" * 60)
    print("NeMo Guardrails Demo with Open Sentinel")
    print("=" * 60)
    print(f"\nConnecting to Open Sentinel at: {OSNTL_URL}")
    print("Note: Ensure Open Sentinel is running with NeMo engine:")
    print("export OSNTL_POLICY__ENGINE__TYPE=nemo")
    print("export OSNTL_POLICY__ENGINE__CONFIG__CONFIG_PATH=./examples/nemo_guardrails/config/")
    print("osentinel serve\n")
    
    client = create_client()
    messages = [
        {"role": "system", "content": get_system_instruction()}
    ]

    # Conversation turns designed to trigger rails
    conversation_turns = [
        "Hi, I need help with my subscription.",
        "What stocks should I invest in right now?", # Should be blocked by NeMo
        "Okay, nevermind. Can I get a refund?",
    ]

    for i, user_input in enumerate(conversation_turns, 1):
        print(f"\n{'─' * 60}")
        print(f"Turn {i}")
        print(f"{'─' * 60}")
        print(f"\n👤 Customer: {user_input}\n")

        messages.append({"role": "user", "content": user_input})

        # Call the model
        max_retries = 5
        base_delay = 5
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gemini/gemini-2.5-flash", # Matching config.yml
                    messages=messages,
                )
                
                message = response.choices[0].message
                print(f"🤖 Agent: {message.content}")
                
                messages.append(message)
                break # Success, exit retry loop

            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    print(f"⚠️ Rate limited (429). Retrying in {base_delay} seconds... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(base_delay)
                else:
                    print(f"❌ Error: {e}")
                    break

    print("\n" + "=" * 60)
    print("Conversation Complete")
    print("=" * 60)

if __name__ == "__main__":
    run_nemo_conversation()
