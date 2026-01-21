import requests

# Use the thread we created
tid = 'a63a321c-205a-402d-9c8a-e271d4660aaa'

# Send a very direct question
print("Sending question from Alex...")
r = requests.post(
    f'http://localhost:8080/threads/{tid}/messages',
    json={
        'message': 'Who said they love Python? And who prefers JavaScript?',
        'primary_user': 'Alex',
        'conversation_source': 'discord',
        'metadata': {'username': 'Alex'}
    }
)

print("\nNova's response:")
print(r.json()['assistant_message']['content'])
