import requests
import json

# Use the thread we just created
tid = 'a63a321c-205a-402d-9c8a-e271d4660aaa'

# Get all messages
msgs = requests.get(f'http://localhost:8080/threads/{tid}/messages').json()

print('Messages in thread:')
print('='*80)
for m in msgs:
    metadata = m.get('metadata', {})
    username = metadata.get('username', 'N/A')
    print(f"\n{m['role'].upper()} (username in metadata: {username}):")
    print(f"  {m['content'][:150]}...")
print('='*80)

# Test if the conversation history now shows usernames
print("\nSending a follow-up question from Alex...")
r = requests.post(
    f'http://localhost:8080/threads/{tid}/messages',
    json={
        'message': 'Can you summarize what each person said so far?',
        'primary_user': 'Alex',
        'conversation_source': 'discord',
        'metadata': {'username': 'Alex'}
    }
)

print("\nNova's response:")
print(r.json()['assistant_message']['content'])
