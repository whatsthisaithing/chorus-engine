import requests

print("Creating fresh Discord conversation...")
conv = requests.post(
    'http://localhost:8080/conversations',
    json={
        'character_id': 'nova',
        'title': 'Fresh Multi-User Test',
        'source': 'discord'
    }
).json()

threads = requests.get(f'http://localhost:8080/conversations/{conv["id"]}/threads').json()
tid = threads[0]['id']
print(f"Created thread: {tid}\n")

# Alex speaks
print("Alex: Hey Nova, I'm a Python developer!")
r1 = requests.post(
    f'http://localhost:8080/threads/{tid}/messages',
    json={
        'message': 'Hey Nova, I am a Python developer!',
        'primary_user': 'Alex',
        'conversation_source': 'discord',
        'metadata': {'username': 'Alex', 'platform': 'discord'}
    }
)
print(f"Nova: {r1.json()['assistant_message']['content'][:150]}...\n")

# Sarah speaks  
print("Sarah: I work with JavaScript mostly")
r2 = requests.post(
    f'http://localhost:8080/threads/{tid}/messages',
    json={
        'message': 'I work with JavaScript mostly',
        'primary_user': 'Sarah',
        'conversation_source': 'discord',
        'metadata': {'username': 'Sarah', 'platform': 'discord'}
    }
)
print(f"Nova: {r2.json()['assistant_message']['content'][:150]}...\n")

# Alex asks about both
print("Alex: Nova, can you tell me who uses which language?")
r3 = requests.post(
    f'http://localhost:8080/threads/{tid}/messages',
    json={
        'message': 'Nova, can you tell me who uses which language?',
        'primary_user': 'Alex',
        'conversation_source': 'discord',
        'metadata': {'username': 'Alex', 'platform': 'discord'}
    }
)
print(f"Nova: {r3.json()['assistant_message']['content']}")
