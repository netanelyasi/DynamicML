import openai
import os
import json
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from fastapi import HTTPException
from app.logger import logger

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1))
def _call_openai(prompt: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            timeout=30
        )
        return response['choices'][0]['message']['content']
    
    except openai.error.AuthenticationError:
        raise HTTPException(401, "Invalid OpenAI API key")
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise HTTPException(500, f"AI service error: {str(e)}")

def suggest_features(summary: list, correlation: dict) -> str:
    prompt = f"""בצע ניתוח מקצועי של הנתונים הבאים בעברית:
    
נתונים סטטיסטיים:
{json.dumps(summary, ensure_ascii=False, indent=2)}

מטריצת קורלציה:
{json.dumps(correlation, ensure_ascii=False, indent=2)}

יש לענות בעברית עם:
1. המלצה לעמודת יעד מתאימה
2. 3-5 עמודות מומלצות כפיצ'רים
3. התרעות על בעיות נתונים אפשריות"""

    return _call_openai(prompt)

def explain_prediction(input_data: dict, prediction) -> str:
    prompt = f"""הסבר בעברית את תוצאת החיזוי הבאה:
    
קלט:
{json.dumps(input_data, ensure_ascii=False, indent=2)}

תוצאה: {prediction}

ההסבר צריך לכלול:
- פרשנות מקצועית של התוצאה
- אילו פיצ'רים הכי השפיעו
- המלצות פעולה בהתאם לתוצאה"""

    return _call_openai(prompt)
