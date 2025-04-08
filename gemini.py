import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_KEY = os.getenv('GEMINI_KEY')
print(GEMINI_KEY)

# from google import genai

# client = genai.Client(api_key=GEMINI_KEY)

# response = client.models.generate_content(
#     model="gemini-2.0-flash",
#     contents=["How does AI work?"]  # modify this
# )

# print(response.text)