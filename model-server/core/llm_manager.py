from llama_cpp import Llama


class LLMManager:
    def __init__(self):
        self.llm = Llama(model_path="models_file/mistral-7b-instruct-v0.1.Q4_K_M.gguf", n_gpu_layers=40, n_ctx=4096,
                         varbose=False,
                         )
        print("✅ LLM model loaded successfully")

    def get_llm_response(self, user_query: str) -> str:
        response = self.llm.create_completion(
            user_query,
            stream=False,
            temperature=0.7,
            top_p=0.9,
            max_tokens=256
        )

        content = response["choices"][0]["text"].strip()
        
    
        
        
        
        

        return content
    
    def get_llm_response_chat(self, chat_history:list) -> str:
        response = self.llm.create_chat_completion(
            chat_history,
            stream=False,
            temperature=0.7,
            top_p=0.9,
            max_tokens=256
        )

        content = response["choices"][0]["message"]["content"]
        
    
        
        
        
        

        return content
