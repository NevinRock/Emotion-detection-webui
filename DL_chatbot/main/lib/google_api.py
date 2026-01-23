import requests
import json
import os

# API Key provided by user
API_KEY = ''
# Using REST API because google-genai SDK requires Python 3.9+, and this environment is 3.8.
# Model confirmed to be 'models/gemini-2.5-flash'
MODEL_NAME = "gemini-2.5-flash" 
m_name_escaped = "gemini-2.5-flash" # or "models/gemini-2.5-flash"? usually just the ID in the URL, but let's see. 
# The LIST output showed "models/gemini-2.5-flash". 
# The REST URL format: .../models/{modelId}:generateContent
# API docs say modelId usually doesn't include 'models/' prefix in the URL segment if the endpoint is v1beta/models/... 
# but let's try strict ID.

URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"

def gpt_response(prompt_input, history=None):
    """
    Generates a response using Google's Gemini model via REST API.
    (SDK not used due to Python 3.8 constraint)
    """
    try:
        gemini_contents = []
        
        # Convert history
        if history:
            for msg in history:
                role = msg.get("role")
                content = msg.get("content")
                
                # Map roles: 'assistant' -> 'model', 'user' -> 'user'
                if role == "assistant":
                    gemini_role = "model"
                else:
                    gemini_role = "user"
                
                gemini_contents.append({
                    "role": gemini_role,
                    "parts": [{"text": content}]
                })
        
        # Append current prompt
        gemini_contents.append({
            "role": "user",
            "parts": [{"text": prompt_input}]
        })

        print("--- Gemini User Input ---")
        # Try to print just the user text if formatted that way
        if "User input: " in prompt_input:
             # Extract the part after "User input: " and before any subsequent newline or footer
             parts = prompt_input.split("User input: ")
             remaining = parts[-1]
             # In app_gradio: ...\nUser input: {user_text}\nRespond...
             # We want to capture {user_text}
             clean_input = remaining.split("\nRespond")[0].strip()
             print(clean_input)
        else:
             print(prompt_input)
        print("-------------------------")

        payload = {
            "contents": gemini_contents
        }
        
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.post(URL, headers=headers, data=json.dumps(payload))
        
        if response.status_code == 200:
            result = response.json()
            # Extract text
            # Response structure: candidates[0].content.parts[0].text
            try:
                text = result['candidates'][0]['content']['parts'][0]['text']
                return text
            except (KeyError, IndexError) as e:
                return f"Error parsing Gemini response: {result}"
        else:
            return f"Error calling Gemini API: {response.status_code} - {response.text}"

    except Exception as e:
        return f"Error communicating with Gemini (REST): {e}"

if "__main__" == __name__:
    print(gpt_response("Explain how AI works in a few words"))

