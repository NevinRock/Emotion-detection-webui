from openai import OpenAI

client = OpenAI(api_key="")

def gpt_response(prompt_input, history=None):
    if history is None:
        history = []
        
    messages = list(history)
    messages.append({"role": "user", "content": prompt_input})

    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error communicating with OpenAI: {e}"

if "__main__" == __name__:
    print(gpt_response("Explain what a CNN is in one sentence for a beginner."))

