import os
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

if __name__ == "__main__":
    # List of required environment variables
    required_vars = [
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "OPENAI_API_KEY"  # if you're using OpenAI for MoM
    ]

    # Check for any missing variables
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        print(f"❌ Error: Missing required environment variables: {', '.join(missing)}")
        print("Please set them in a .env file or export them in your environment.")
        exit(1)

    # Start FastAPI app
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
