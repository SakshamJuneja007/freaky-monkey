from fastapi import FastAPI, Request, Query
from fastapi.responses import PlainTextResponse
import json

app = FastAPI()

VERIFY_TOKEN = "deimos_call_test_2026"


@app.get("/webhook")
async def verify_webhook(
    mode: str | None = Query(default=None, alias="hub.mode"),
    verify_token: str | None = Query(default=None, alias="hub.verify_token"),
    challenge: str | None = Query(default=None, alias="hub.challenge"),
):
    print("\n--- WEBHOOK VERIFICATION ---")
    print("mode:", mode)
    print("verify_token:", verify_token)
    print("challenge:", challenge)

    if mode == "subscribe" and verify_token == VERIFY_TOKEN:
        print("Verification SUCCESS")
        return PlainTextResponse(challenge or "")

    print("Verification FAILED")
    return PlainTextResponse("Verification failed", status_code=403)


@app.post("/webhook")
async def receive_webhook(request: Request):
    data = await request.json()

    print("\n" + "=" * 70)
    print("WHATSAPP WEBHOOK EVENT RECEIVED")
    print("=" * 70)

    print(json.dumps(data, indent=2))

    # Specifically identify Calling events
    for entry in data.get("entry", []):
        for change in entry.get("changes", []):
            field = change.get("field")
            print(f"\nWebhook field: {field}")

            if field == "calls":
                print("\n🔥 INCOMING WHATSAPP CALL EVENT 🔥")
                print(json.dumps(change, indent=2))

    print("=" * 70 + "\n")

    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )