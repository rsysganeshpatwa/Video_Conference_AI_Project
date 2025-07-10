from llama_cpp import Llama

class LLMManager:
    def __init__(self, model_path: str = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",n_ctx=32000, n_gpu_layers=20):
        print("🧠 Loading LLM...")
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx,   n_gpu_layers=-1, use_mlock=True, use_mmap=True, logits_all=False, verbose=True)

    def get_llm_response(self, user_query: str, stream:bool = False) -> str:
        print("\n💬 Generating response from Mistral LLM...")

        system_prompt = (
            "You are a helpful, friendly, and conversational AI voice assistant. "
            "Speak casually and naturally, using short phrases like 'okay', 'sure', 'thanks', or 'I'll do that'. "
            "Always respond clearly in 1–2 short sentences."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        response = self.llm.create_chat_completion(
            messages,
            temperature=0.5,
            top_p=0.9,
            max_tokens=128,
        )
        return response["choices"][0]["message"]["content"]



# Example usage:
if __name__ == "__main__":
     llm_manager = LLMManager()
     response = llm_manager.get_llm_response("What is the capital of France?")
     print("LLM Response:", response)