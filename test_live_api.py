import json
import urllib.request

payload = {
    "user_id": "demo-user",
    "password": "demo-pass",
    "conversation": [
        {"role": "user", "content": "I had a great day at work and I am excited about the project."},
        {"role": "assistant", "content": "That sounds wonderful. I am happy to hear it."},
        {"role": "user", "content": "I need to buy groceries later this evening."},
    ],
}
req = urllib.request.Request(
    "http://127.0.0.1:8010/upload",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req) as response:
    print(response.read().decode("utf-8"))
