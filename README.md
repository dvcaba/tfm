# Tweet Sentiment Analysis Agent

This project exposes a FastAPI service that analyzes tweet sentiment using a fine-tuned model and answers user questions through an agent built with LangGraph. Responses are generated with Anthropic's Claude models.

## Configuration

Set an Anthropic API key in the environment so the agent can call Claude for both answering questions and detecting intent:

```bash
export ANTHROPIC_API_KEY=your-key-here
```

Without this variable the API will fail when attempting to contact Anthropic.

## Running the API

Install the requirements and run the FastAPI application:

```bash
pip install -r requirements.txt
python main.py
```

Then POST questions to `http://localhost:8000/ask` with a JSON payload `{"question": "<your question>"}`.
If the question is `"salir"` or `"no"` the service will return a farewell and no further processing will occur.
