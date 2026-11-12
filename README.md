# Agentic Chatbot with LangGraph

## Run locally with Docker

Create a `.env` file in the project directory with the API keys:

```env
OPENROUTER_API_KEY=your-openrouter-api-key
TAVILY_API_KEY=your-tavily-api-key
WEATHERSTACK_API_KEY=your-weatherstack-api-key
ALPHAVANTAGE_API_KEY=your-alphavantage-api-key
```

From the project directory, build and run the image:

```powershell
docker build -t agentic-chatbot:local .
docker run --rm -p 8501:8501 --env-file .env agentic-chatbot:local
```

Open `http://localhost:8501`. Keep `.env` private; it is excluded from Git and Docker build contexts.

## Deploy on Render

1. In Render, create a **Web Service** from this GitHub repository and select the `main` branch.
2. Choose the **Docker** runtime and use the repository's `Dockerfile`.
3. Add these environment variables in the Render service settings:
   - `OPENROUTER_API_KEY`
   - `TAVILY_API_KEY`
   - `WEATHERSTACK_API_KEY`
   - `ALPHAVANTAGE_API_KEY`
4. If using the GitHub Actions deploy workflow, turn off Render's automatic deploys, create a Render deploy hook, and save its URL as the GitHub Actions secret `RENDER_DEPLOY_HOOK_URL`.
5. Push to `main`. GitHub Actions checks Python syntax, then calls the Render deploy hook. Render builds and deploys the Docker image.

LangSmith variables (`LANGSMITH_TRACING`, `LANGSMITH_ENDPOINT`, `LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT`) are optional. No AWS or Docker Hub credentials are needed.
