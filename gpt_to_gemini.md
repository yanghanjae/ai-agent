# GPT → Gemini 변환 요약

| GPT | Gemini |
|---|---|
| `OpenAI(api_key=...)` | `genai.configure(api_key=...)` |
| `client.chat.completions.create()` | `model.generate_content()` |
| `model="gpt-4o"` | `model_name='gemini-2.0-flash'` |
| `{"role": "system", "content": ...}` | `system_instruction=...` (모델 생성 시) |
| `{"role": "assistant", "content": ...}` | `{"role": "model", "parts": [...]}` |
| `{"role": "user", "content": ...}` | `{"role": "user", "parts": [...]}` |
| `temperature=0.9` | `generation_config={"temperature": 0.9}` |
| `response.choices[0].message.content` | `response.text` |

## 한 줄 요약
`assistant→model`, `content→parts`, `system→system_instruction`, `응답→response.text`
