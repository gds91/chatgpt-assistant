# ChatGPT as a virtual assistant

## Instructions
- Install dependencies. Use pip `install -r requirements.txt`.  
  Alternatively, use `python install-dependencies` to install all requirements and accompanying dependencies.
- You will need to provide an OpenAI api key and an Azure congnition api key in either self-created `config.py` file at the root level or in `environment variables`.  
  The values required are `OPENAI_API_KEY`, `AZURE_API_KEY` and `AZURE_REGION` for OpenAi and Azure respectively.
- You will need to install JDK and FFMPEG on top of that. Make sure your paths are properly set up for both of them.
- Run the assistant with `python assistant.py`