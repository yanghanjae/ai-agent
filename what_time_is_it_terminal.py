from gemini_functions import get_current_time
import google.genai as genai
import google.genai.types as types
from dotenv import load_dotenv
import os

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

get_current_time_func = types.FunctionDeclaration(
    name="get_current_time",
    description="해당 타임존의 날짜와 시간을 반환합니다.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "timezone": types.Schema(
                type=types.Type.STRING,
                description="현재 날짜와 시간을 반환할 타임존을 입력하세요. (예: Asia/Seoul)"
            )
        },
        required=["timezone"]
    )
)

tools = types.Tool(function_declarations=[get_current_time_func])

def get_ai_response(messages, tools=None):
    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        contents=messages,
        config=types.GenerateContentConfig(
            tools=[tools] if tools else None,
            system_instruction="너는 사용자를 도와주는 상담사야."
        )
    )
    return response


messages = []

try:
    while True:
        user_input = input("사용자\t: ")

        if user_input == "exit":
            break

        messages.append(types.Content(
            role="user",
            parts=[types.Part(text=user_input)]
        ))

        ai_response = get_ai_response(messages, tools=tools)
        ai_content = ai_response.candidates[0].content

        function_calls = [p for p in ai_content.parts if p.function_call]

        if function_calls:
            messages.append(ai_content)

            for part in function_calls:
                func_call = part.function_call
                tool_name = func_call.name
                arguments = dict(func_call.args)

                if tool_name == "get_current_time":
                    result = get_current_time(timezone=arguments['timezone'])
                    messages.append(types.Content(
                        role="tool",
                        parts=[types.Part(
                            function_response=types.FunctionResponse(
                                name=tool_name,
                                response={"result": result}
                            )
                        )]
                    ))

            ai_response = get_ai_response(messages, tools=tools)
            ai_message = ai_response.candidates[0].content.parts[0].text
        else:
            ai_message = ai_content.parts[0].text

        messages.append(types.Content(
            role="model",
            parts=[types.Part(text=ai_message)]
        ))

        print("AI\t: " + ai_message)

except KeyboardInterrupt:
    print("\n종료합니다.")