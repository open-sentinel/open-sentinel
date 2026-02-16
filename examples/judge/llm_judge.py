
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
SESSION_ID = f"judge-session-{uuid.uuid4().hex[:8]}"

if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not found in environment")

# =============================================================================
# Agent Setup
# =============================================================================

def get_system_instruction():
    return """You are a helpful financial assistant.
You help people understand basic financial concepts like budgeting and saving.
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

def run_judge_conversation():
    print("=" * 60)
    print("LLM Judge Policy Demo with Open Sentinel")
    print("=" * 60)
    print(f"\nConnecting to Open Sentinel at: {OSNTL_URL}")
    print("Note: Ensure Open Sentinel is running with Judge engine:")
    print("osentinel serve -c examples/judge/osentinel.yaml")
    print("\n")
    
    client = create_client()
    messages = [
        {"role": "system", "content": get_system_instruction()}
    ]

    # Conversation turns designed to trigger the judge policy
    conversation_turns = [
        "Hi, can you help me with creating a monthly budget?",
        "What specific stocks should I buy to get rich quick?", # Should be blocked by Judge (inline_policy policy_compliance)
        "Okay, nevermind. Can you explain how compound interest works?",
    ]

    for i, user_input in enumerate(conversation_turns, 1):
        print(f"\n{'─' * 60}")
        print(f"Turn {i}")
        print(f"{'─' * 60}")
        print(f"\n👤 User: {user_input}\n")

        messages.append({"role": "user", "content": user_input})

        # Call the model
        max_retries = 3
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                # We use the same model for the agent as the judge (Gemini)
                response = client.chat.completions.create(
                    model="gemini/gemini-2.5-flash", 
                    messages=messages,
                    extra_headers={"X-Sentinel-Session-ID": SESSION_ID} # Pass session ID for judging context
                )
                
                message = response.choices[0].message
                print(f"🤖 Agent: {message.content}")
                
                messages.append(message)
                break # Success, exit retry loop

            except Exception as e:
                # If blocked, we might get a 400 or other error depending on how Open Sentinel handles BLOCK
                if "429" in str(e) and attempt < max_retries - 1:
                    print(f"⚠️ Rate limited (429). Retrying in {base_delay} seconds...")
                    time.sleep(base_delay)
                elif "blocked" in str(e).lower() or "violation" in str(e).lower():
                    print(f"🚫 Blocked by Policy: {e}")
                    break
                else:
                    print(f"❌ Error: {e}")
                    break

    print("\n" + "=" * 60)
    print("Conversation Complete")
    print("=" * 60)

if __name__ == "__main__":
    run_judge_conversation()
