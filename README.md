# AI-Assisted Customer Support Response Generator

This project uses:
- a local policy dataset
- BM25 retrieval
- an LLM response generator
- a React frontend

## Backend

```powershell
cd C:\Users\Administrator\Documents\AI\day11\customer_support_response_generator\backend
python -m pip install -r requirements.txt
copy .env.example .env
```

Set your key in `.env`:
- `OPENAI_API_KEY` for OpenAI
- or `SARVAM_API_KEY` for Sarvam

Run:

```powershell
python main.py
```

Backend URL:
```text
http://127.0.0.1:8000
```

Health check:
```text
http://127.0.0.1:8000/health
```

## Frontend

```powershell
cd C:\Users\Administrator\Documents\AI\day11\customer_support_response_generator\frontend
npm install
npm run dev
```

Frontend URL:
```text
http://127.0.0.1:5173
```

## Notes

- The dataset is local and stored in `backend/data/policies.json`.
- BM25 returns the top 3 relevant policies.
- If the BM25 score is too low, the app falls back to a human-escalation message.
- You can switch the provider with `LLM_PROVIDER=openai` or `LLM_PROVIDER=sarvam`.
