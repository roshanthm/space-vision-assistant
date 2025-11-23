import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def explain_result(label):
    prompt = f"Explain in simple terms what a {label} is in astronomy."

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except:
        return "Explanation not available (Gemini error)."
