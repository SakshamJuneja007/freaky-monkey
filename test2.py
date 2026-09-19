import os
import requests
from dotenv import load_dotenv

load_dotenv()

APP_ID = "2202427980234937"
TOKEN = "EAAUdY9aCQvgBSizYuZAGQWDWbhccS2KCRtFSKV0GgQPqw9G4jeRZCbZAo5fj6AC4k2tAXvQ5ZCqZBZAJTMbAoZBUF3JEUmkamB7R51e7prhMSeX7R0BhLCZAG94Ev7tIwatIXkY1hwS1Uz2gcyTK9kzsoZBN5MUAVyZB5IG9nNmd7FbjqZCBX0pbMGazVeNJQCLHwZDZD"

url = f"https://graph.facebook.com/v25.0/{APP_ID}/subscriptions"

r = requests.get(
    url,
    headers={"Authorization": f"Bearer {TOKEN}"},
    timeout=30
)

print("HTTP:", r.status_code)
print(r.text)