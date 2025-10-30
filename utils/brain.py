import os
import time


def decoder_for_response(args, engine, messages, max_new_tokens, file_list=None):
    if "gpt4o" in engine:
        import openai
        from openai import OpenAI

        # configure your openai key by `export OPENAI_API_KEY=""` in command line
        api_key = os.environ["OPENAI_API_KEY"]
        client = OpenAI(api_key=api_key)

        while True:
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-2024-05-13",
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=0,
                    seed=args.seed,
                    n=1,
                )
                predicted_answers = response.choices[0].message.content
                break
            except openai.RateLimitError as e:
                print("Rate limit reached, waiting for 1 hour")
                time.sleep(3600)  # Wait for 1 hour (3600 seconds)
                continue
            except Exception as e:
                print(e)
                print("pausing")
                time.sleep(1)
                continue
    elif "gemini" in engine:
        import google.generativeai as genai
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        # configure your openai key by `export GEMINI_API_KEY=""` in command line
        api_key = os.environ["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

        def clear_gemini_img_files(verbose=True):
            for f in genai.list_files():
                myfile = genai.get_file(f.name)
                myfile.delete()
                if verbose:
                    print("Deleted", f.name)

        # clear_gemini_img_files()

        while True:
            predicted_answers = "no response"
            try:
                generation_config = {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_output_tokens": max_new_tokens,
                }
                safety_settings = [
                    {
                        "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        "threshold": HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                        "threshold": HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        "threshold": HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        "threshold": HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        "threshold": HarmBlockThreshold.BLOCK_NONE,
                    },
                ]
                response = model.generate_content(
                    messages,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                )
                assert hasattr(response, "text")
                # predicted_answers = response.text
                try:
                    # Check if 'candidates' list is not empty
                    if response.candidates:
                        # Access the first candidate's content if available
                        if response.candidates[0].content.parts:
                            predicted_answers = (
                                response.candidates[0].content.parts[0].text
                            )
                        else:
                            print("No generated text found in the candidate.")
                    else:
                        print("No candidates found in the response.")
                except (AttributeError, IndexError) as e:
                    print("Error:", e)
                time.sleep(1)
                break
            except Exception as e:
                predicted_answers = f"[ERROR] gemini failed: {e}"
                print(f"[ERROR] gemini failed: {e}")
                time.sleep(1)
                break
        for f in file_list:
            f.delete()

    return predicted_answers
