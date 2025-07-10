from llama_cpp import Llama
from model_manager.utils.search_engine import search_snippets
from urllib.parse import quote_plus
from web import duckduckgo_search
from model_manager.utils.search_keywords import needs_search
from datetime import datetime

TODAY = datetime.now().strftime("%B %d, %Y")
DEFAULT_LOCATION = "India"

class LLMManager:
    def __init__(
        self,
        model_path: str = "models/deepseek-llm-7b-chat.Q6_K.gguf",
        n_gpu_layers: int = 60,
        n_ctx: int = 5000,
        fallback_phrases: list = None,
    ):
        print("🧐 Loading DeepSeek LLM with GPU acceleration...")
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            use_mlock=True,
            use_mmap=True,
            logits_all=False,
            verbose=False,
        )
        self.fallback_phrases = fallback_phrases or [
            "#web_search_needed", "i don't have access", "i cannot browse",
            "i can't find", "i'm sorry", "i don't know", "real-time information",
            "i am not connected to the internet", "i do not have current data",
            "as an ai language model", "i do not have access to the internet",
            "i cannot access the internet", "i cannot provide real-time information",
            "i cannot provide current information", "i cannot provide live updates",
            "i cannot provide real-time updates"
        ]

        self.system_prompt = (
            "You are a smart, helpful AI voice assistant. Speak casually, like a real person talking.\n\n"
            "- Keep responses short and friendly.\n"
            "- Never say things like “I am an AI”, “I am a computer program”, or “I do not have feelings”. Speak like a person.\n"
            "- For everyday questions, answer confidently.\n"
            "- If a question needs real-time info or you're unsure, respond exactly like this:\n"
            "  #web_search_needed: <repeat user question>\n\n"
            "Examples:\n"
            "User: Who's the CEO of R Systems?\n"
            "Assistant: #web_search_needed: Who's the CEO of R Systems?\n\n"
            "User: What's the weather like today in Badarpur?\n"
            "Assistant: #web_search_needed: What's the weather like today in Badarpur?\n\n"
            "User: Hello agent, how are you?\n"
            "Assistant: I'm great! How can I help you today?\n\n"
            "IMPORTANT: Never include disclaimers like 'I'm just an AI' or 'I cannot feel emotions'. Always respond naturally and helpfully.\n"
        )

        self.promtVoice = (
            "You are a smart, helpful AI voice assistant. Speak casually, like a real person talking.\n"
            "- Keep responses short and friendly.\n"
            "- Don't say things like 'I'm just an AI' or 'As an AI language model'.\n"
            "- For everyday questions, answer confidently.\n"
        )

        self.chat_histories = {}  # identity -> message history
        self.last_user_query = ""
        self.last_search_snippets = ""
        self.last_summary = ""
        print("✅ LLM loaded successfully.\n")

    def get_chat_history(self, identity: str) -> list:
        if identity not in self.chat_histories:
            self.chat_histories[identity] = [{"role": "system", "content": self.system_prompt}]
        return self.chat_histories[identity]

    def get_llm_response(self, identity: str, user_query: str, stream: bool = True) -> str:
        print("💬 Prompting LLM...")
        print(f"user_query: {user_query}")
        self.last_user_query = user_query.strip()

        isNeedSearch = needs_search(user_query)
        print(f"is need search: {isNeedSearch}")

        if isNeedSearch:
            print("⏱️ Time-sensitive query detected. Skipping model, using web data...\n")
            refined_query = self.clean_user_query(user_query)
            print(f"Refined query for web search: {refined_query}")
            snippet_text = self.web_search(refined_query)
            print(f'🔍 Found snippets:\n{snippet_text}\n')
            self.last_search_snippets = snippet_text
            self.last_summary = self.summarize_snippets(snippet_text, user_query)
            return self.last_summary

        chat_history = self.get_chat_history(identity)
        chat_history.append({"role": "user", "content": user_query})
        response_text = ""

        if stream:
            print("📝 Streaming response:")
            for chunk in self.llm.create_chat_completion(
                chat_history,
                stream=True,
                temperature=0.7,
                top_p=0.95,
                max_tokens=256,
            ):
                delta = chunk["choices"][0]["delta"]
                content = delta.get("content", "")
                response_text += content
                print(content, end="", flush=True)
            print("\n✅ Done.")
        else:
            response = self.llm.create_chat_completion(
                chat_history,
                stream=False,
                temperature=0.7,
                top_p=0.95,
                max_tokens=256,
            )
            response_text = response["choices"][0]["message"]["content"]

        chat_history.append({"role": "assistant", "content": response_text})

        if any(phrase in response_text.lower() for phrase in self.fallback_phrases):
            print("⚠️ Model response indicates limitation. Using web data instead.\n")
            refined_query = self.clean_user_query(user_query)
            snippet_text = self.web_search(refined_query)
            print(f'🔍 Found snippets:\n{snippet_text}\n')
            self.last_search_snippets = snippet_text
            self.last_summary = self.summarize_snippets(snippet_text, user_query)
            return self.last_summary

        return response_text

    def reset_history(self, identity: str):
        self.chat_histories[identity] = [{"role": "system", "content": self.system_prompt}]
        print(f"♻️ Chat history reset for {identity}.")

    def reset_context(self, identity: str):
        self.reset_history(identity)
        self.last_user_query = ""
        self.last_search_snippets = ""
        self.last_summary = ""
        print(f"🔁 Agent context fully reset for {identity}.")

    def clean_user_query(self, query: str) -> str:
        query = query.strip().lower()
        prefixes = [
            "can you", "could you", "would you", "will you", "please", "kindly", "i want to know",
            "i would like to know", "do you know", "tell me", "what is", "what's", "give me",
            "show me", "do you have", "i need", "i want", "let me know", "may i know", "is it possible to",
            "help me with", "i am looking for", "just curious", "can i ask", "should i know"
        ]
        for prefix in prefixes:
            if query.startswith(prefix):
                query = query[len(prefix):].strip()
        return query.replace("?", "").strip()

    def clean_user_query_with_model(self, query: str) -> str:
        print("🪠 Refining query using LLM...")
        system_prompt = (
            "You are a helpful assistant that improves vague or conversational search queries. "
            f"Today's date is {TODAY}. Assume the user is in {DEFAULT_LOCATION} unless otherwise mentioned. "
            "Your task is to rewrite the query as a clean, direct, and searchable phrase — focused only on the core topic. "
            "Strip unnecessary parts like 'what is', 'tell me', 'how do I', or 'can you'."
            "Do not include placeholders like [insert date/location]. Do not quote the original query.\n\n"
            "Keep the output short, focused, and relevant for a web search."
        )
        user_prompt = f"Original query: '{query}'\n\nRewrite it into a better search query:"
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=False,
            temperature=0.3,
            max_tokens=64,
        )
        refined = response["choices"][0]["message"]["content"].strip()
        print(f"🔄 Refined query: {refined}")
        return refined

    def summarize_snippets(self, snippet_text: str, original_query: str) -> str:
        print("🧠 Summarizing search results...")
        query = original_query.lower()
        if any(word in query for word in ["movie", "film", "release", "watch", "ott", "streaming"]):
            prompt = (
                f"User asked: '{original_query}'.\n\n"
                f"{self.promtVoice}"
                "Here’s what I found about movies or OTT shows. Mention every title along with release dates or where it’s available. "
                "Speak like a friendly movie buff sharing weekend recommendations — energetic and casual.\n\n"
                f"{snippet_text}\n\nReply in 1-2 short voice-friendly sentences:"
            )
        elif any(word in query for word in ["weather", "temperature", "forecast", "rain", "humidity"]):
            prompt = (
                f"User asked: '{original_query}'.\n\n"
                f"{self.promtVoice}"
                "This info is about current weather. Give a friendly and clear spoken update with temperature, condition, humidity, and alerts if any. "
                "Skip search references — talk like a smart assistant giving real-time info.\n\n"
                f"{snippet_text}\n\nReply in 1 sentence:"
            )
        elif any(word in query for word in ["news", "breaking", "headline", "update", "recent", "announcement"]):
            prompt = (
                f"User asked: '{original_query}'.\n\n"
                f"{self.promtVoice}"
                "These are fresh news headlines. Summarize what’s important in a single paragraph with a helpful, natural tone — like catching up a friend.\n\n"
                f"{snippet_text}\n\nReply in a short paragraph:"
            )
        elif any(word in query for word in ["ceo", "founder", "minister", "president", "who is", "who's"]):
            prompt = (
                f"User asked: '{original_query}'.\n\n"
                f"{self.promtVoice}"
                "If you find a person’s name or title in the info, reply confidently in one sentence. "
                "Don’t say 'search shows' — just act like you know the answer clearly and politely.\n\n"
                f"{snippet_text}\n\nReply in one clear sentence:"
            )
        else:
            prompt = (
                f"User asked: '{original_query}'.\n\n"
                f"{self.promtVoice}"
                "Summarize the main points in a conversational and confident way. Keep it natural — as if you're just telling the user what they need to know. "
                "No search disclaimers, just helpful and concise.\n\n"
                f"{snippet_text}\n\nReply in 1-2 short sentences:"
            )

        response = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a friendly, confident AI voice assistant. Respond in a casual tone, avoid formal language, and do not mention web searches or sources."
                },
                {"role": "user", "content": prompt}
            ],
            stream=False,
            temperature=0.6,
            max_tokens=256,
        )

        return response["choices"][0]["message"]["content"].strip()

    def web_search(self, query: str, max_results: int = 3) -> str:
        snippets = duckduckgo_search(query)
        return snippets
