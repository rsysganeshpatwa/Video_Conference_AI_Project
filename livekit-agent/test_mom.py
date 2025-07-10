from llama_cpp import Llama
import os

MODEL_DIR = "/home/vikas/POC-VideoConferenceTool/livekit-agent/models"

# List your models here, key = label, value = model file name
MODELS = {
    "openchat-3.5-4k": "openchat-3.5-0106.Q4_K_M.gguf",
    "mistral-4k": "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    # add more models if you want
}

TRANSCRIPT = """
Vikas Rana2kDQc: q sur Integration open API integration kanda baakiye saati hum multi language support or speech revanization par bhi karm karein Sounds good. This is the front end and back end two are milker and smoothly come back end is monitoring and focusing on scaling and you are having doneloud threats Bye. kiya okay
Ganeshd9oMm: Arabha setups GOOD MORNING ING tvE거Pos select Screws Uŕ Dets great, what is the best performance of API? Any latency issues? Excellent, now next step, what is the next step? The next step is pending. I am working on UI improvement with the front-end team, like real-time management, mood, features or dynamic language switching. going to try, provide or define a video editing ensure that perfect our action item is because api integration multi language beacon model to complete karnai karnai kanesh print 10 my language switching participant and audio processing Sudharnai dono performance and stress testing plan or existing. Thank you控制
"""

PROMPT_TEMPLATE = """
You are an assistant that generates detailed Minutes of Meeting (MoM) from transcripts.
Split the output into two sections: 'Key Discussion Points' and 'Action Items'.
Each Action Item should be specific, assigned (if possible), and time-bound.

Transcript:
{transcript}

Minutes of Meeting:
"""

def generate_mom_with_model(model_path, prompt, max_tokens=128):
    llm = Llama(model_path=model_path, n_gpu_layers=32, n_ctx=4096, verbose=True)
    response = llm.create_completion(prompt=prompt, max_tokens=max_tokens)
    return response['choices'][0]['text'].strip()

def main():
    prompt = PROMPT_TEMPLATE.format(transcript=TRANSCRIPT)
    
    for label, model_file in MODELS.items():
        full_path = os.path.join(MODEL_DIR, model_file)
        if not os.path.exists(full_path):
            print(f"⚠️ Model not found: {full_path}. Skipping {label}")
            continue
        
        print(f"\n--- Running model: {label} ---")
        try:
            mom = generate_mom_with_model(full_path, prompt)
            print(f"📝 MoM from {label}:\n{mom}\n{'='*50}")
        except Exception as e:
            print(f"❌ Error with model {label}: {e}")

if __name__ == "__main__":
    main()
