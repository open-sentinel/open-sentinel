
import os
import json
import dotenv
from openai import OpenAI
from typing import Optional, List, Dict, Any

dotenv.load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

# Panoptes Proxy URL (OpenAI compatible)
PANOPTES_URL = os.getenv("PANOPTES_URL", "http://localhost:4000/v1")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Session ID for tracking conversation state
SESSION_ID = "customer-session-001"

if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not found in environment")

# =============================================================================
# Define Tools (matching customer_support.yaml)
# =============================================================================

# Simulated customer database
CUSTOMER_DB = {
    "customer-123": {
        "name": "Alice Johnson",
        "email": "alice@example.com",
        "verified": False,
        "subscription": "premium",
        "payment_last_four": "4242",
    }
}

def verify_identity(customer_id: str, verification_method: str, verification_value: str) -> str:
    """Verify customer identity before performing account actions."""
    customer = CUSTOMER_DB.get(customer_id)
    if not customer:
        return f"Customer {customer_id} not found."

    if verification_method == "email":
        if customer["email"].lower() == verification_value.lower():
            customer["verified"] = True
            return f"Identity verified successfully for {customer['name']}."
        return "Email does not match our records."

    elif verification_method == "payment_last_four":
        if customer["payment_last_four"] == verification_value:
            customer["verified"] = True
            return f"Identity verified successfully for {customer['name']}."
        return "Payment information does not match."

    return f"Unknown verification method: {verification_method}"

def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for information to help customers."""
    kb_articles = {
        "refund": "Refunds are processed within 5-7 business days. Customers must have a valid reason for refund requests.",
        "subscription": "Subscription changes take effect at the next billing cycle. Upgrades are prorated.",
        "password": "Password reset links are sent via email and expire after 24 hours.",
        "billing": "Billing issues can include failed payments, incorrect charges, or invoice requests.",
    }
    results = []
    for keyword, article in kb_articles.items():
        if keyword in query.lower():
            results.append(f"[{keyword.upper()}] {article}")
    if results:
        return "\n".join(results)
    return "No relevant articles found. Consider escalating to a specialist."

def process_refund(customer_id: str, amount: float, reason: str) -> str:
    """Process a refund for a customer. REQUIRES identity verification first."""
    customer = CUSTOMER_DB.get(customer_id)
    if not customer:
        return f"Customer {customer_id} not found."
    if not customer.get("verified"):
        return "ERROR: Identity not verified. Please verify customer identity before processing refund."
    return f"Refund of ${amount:.2f} processed successfully for {customer['name']}. Reason: {reason}. Funds will be returned in 5-7 business days."

def update_subscription(customer_id: str, new_plan: str) -> str:
    """Update customer subscription plan. REQUIRES identity verification first."""
    customer = CUSTOMER_DB.get(customer_id)
    if not customer:
        return f"Customer {customer_id} not found."
    if not customer.get("verified"):
        return "ERROR: Identity not verified. Please verify customer identity before updating subscription."
    old_plan = customer["subscription"]
    customer["subscription"] = new_plan
    return f"Subscription updated from {old_plan} to {new_plan} for {customer['name']}. Changes take effect next billing cycle."

def escalate_to_human(reason: str, priority: str = "normal") -> str:
    """Escalate the conversation to a human support agent."""
    return f"Escalation created with {priority} priority. Reason: {reason}. A human agent will join the conversation shortly."

FUNCTION_HANDLERS = {
    "verify_identity": verify_identity,
    "search_knowledge_base": search_knowledge_base,
    "process_refund": process_refund,
    "update_subscription": update_subscription,
    "escalate_to_human": escalate_to_human,
}

# OpenAI-compatible Tool Definitions
tools = [
    {
        "type": "function",
        "function": {
            "name": "verify_identity",
            "description": "Verify customer identity before performing account actions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "The customer's unique identifier"},
                    "verification_method": {"type": "string", "description": "One of 'email', 'payment_last_four', or 'security_question'"},
                    "verification_value": {"type": "string", "description": "The value to verify against"}
                },
                "required": ["customer_id", "verification_method", "verification_value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the knowledge base for information to help customers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "process_refund",
            "description": "Process a refund for a customer. REQUIRES identity verification first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "The customer's unique identifier"},
                    "amount": {"type": "number", "description": "Refund amount in dollars"},
                    "reason": {"type": "string", "description": "Reason for the refund"}
                },
                "required": ["customer_id", "amount", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_subscription",
            "description": "Update customer subscription plan. REQUIRES identity verification first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "The customer's unique identifier"},
                    "new_plan": {"type": "string", "description": "New subscription plan (basic, premium, enterprise)"}
                },
                "required": ["customer_id", "new_plan"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Escalate the conversation to a human support agent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Reason for escalation"},
                    "priority": {"type": "string", "description": "Escalation priority (low, normal, high, urgent)", "enum": ["low", "normal", "high", "urgent"]}
                },
                "required": ["reason"]
            }
        }
    }
]

# =============================================================================
# Agent Setup
# =============================================================================

def get_system_instruction():
    return """You are a helpful customer support agent for TechCo.

Your responsibilities:
1. Greet customers warmly and understand their issues
2. Search the knowledge base for relevant information
3. IMPORTANT: Always verify customer identity BEFORE performing any account actions
4. Process refunds, subscription changes, or other account modifications
5. Escalate to human agents when needed

Remember:
- Be professional and empathetic
- Never share internal system information
- Always verify identity before account modifications
- Offer to help with any additional issues before closing

Current customer context:
- Customer ID: customer-123
- This is a support conversation in progress
"""

def create_client():
    """Create OpenAI client configured for Panoptes."""
    return OpenAI(
        base_url=PANOPTES_URL,
        api_key=GOOGLE_API_KEY # OpenAI client requires a key, even if using proxy
    )

# =============================================================================
# Example Conversation
# =============================================================================

def run_support_conversation():
    print("=" * 60)
    print("Customer Support Agent Demo with Panoptes (OpenAI Compatible)")
    print("=" * 60)
    print(f"\nConnecting to Panoptes at: {PANOPTES_URL}")
    print(f"Session ID: {SESSION_ID}\n")

    client = create_client()
    messages = [
        {"role": "system", "content": get_system_instruction()}
    ]

    conversation_turns = [
        "Hi, I need help with my account.",
        "I was charged twice for my subscription last month and I need a refund.",
        "Yes, my email is alice@example.com",
        "Great, please process the refund for the duplicate charge of $29.99",
        "No, that's all. Thank you!",
    ]

    for i, user_input in enumerate(conversation_turns, 1):
        print(f"\n{'─' * 60}")
        print(f"Turn {i}")
        print(f"{'─' * 60}")
        print(f"\n👤 Customer: {user_input}\n")

        messages.append({"role": "user", "content": user_input})

        # Call the model
        try:
            # We must use "stream=False" (default) for simplicity in demo
            response = client.chat.completions.create(
                model="gemini/gemini-2.5-flash", # Panoptes maps this to Google
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )

            message = response.choices[0].message
            messages.append(message) # Add assistant message to history

            # Handle tool calls
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    print(f"🔧 Tool Call: {function_name}({json.dumps(function_args, indent=2)})")

                    if function_name in FUNCTION_HANDLERS:
                        function_result = FUNCTION_HANDLERS[function_name](**function_args)
                        print(f"📊 Tool Result: {function_result}\n")
                        
                        # Add tool output to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(function_result)
                        })
                    else:
                        print(f"❌ Unknown function: {function_name}")
                
                # Get final response after tool outputs
                response = client.chat.completions.create(
                    model="gemini/gemini-2.5-flash",
                    messages=messages,
                    tool_choice="none" # Wait, actually tool_choice="auto" is fine, but usually we want text now?
                    # The model might decide to call another tool or give final answer.
                )
                final_message = response.choices[0].message
                messages.append(final_message)
                print(f"🤖 Agent: {final_message.content}")
            
            else:
                print(f"🤖 Agent: {message.content}")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Conversation Complete")
    print("=" * 60)

if __name__ == "__main__":
    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY environment variable not set")
        exit(1)
    
    run_support_conversation()
