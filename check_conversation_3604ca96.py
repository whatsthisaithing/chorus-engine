import sqlite3
import json

# Query the specific conversation
conn = sqlite3.connect('j:/Dev/chorus-engine/data/chorus.db')

# Get thread ID
cursor = conn.execute('''
    SELECT id FROM threads 
    WHERE conversation_id = "3604ca96-ad92-4bea-86bc-128bf2819131"
''')
threads = cursor.fetchall()
if threads:
    thread_id = threads[0][0]
    print(f"Thread ID: {thread_id}\n")
else:
    print("No thread found!")
    exit()

# Get messages
cursor = conn.execute('''
    SELECT id, role, content, created_at, metadata 
    FROM messages 
    WHERE thread_id = ? 
    ORDER BY created_at
''', (thread_id,))

print("=" * 120)
print("MARCUS'S CONVERSATION: 3604ca96-ad92-4bea-86bc-128bf2819131")
print("=" * 120)
print()

messages = cursor.fetchall()
for i, row in enumerate(messages, 1):
    msg_id, role, content, created_at, metadata = row
    
    print(f"Message {i}:")
    print(f"  ID: {msg_id}")
    print(f"  Role: {role}")
    print(f"  Content: {content}")
    
    if metadata:
        try:
            metadata_dict = json.loads(metadata)
            print(f"  Metadata:")
            for key, value in metadata_dict.items():
                print(f"    {key}: {value}")
        except:
            print(f"  Metadata (raw): {metadata}")
    
    print()

print(f"Total messages: {len(messages)}")
print()

# Check conversation details
cursor = conn.execute('''
    SELECT id, character_id, title, source 
    FROM conversations 
    WHERE id = "3604ca96-ad92-4bea-86bc-128bf2819131"
''')
conv = cursor.fetchone()
if conv:
    print(f"Conversation Details:")
    print(f"  ID: {conv[0]}")
    print(f"  Character ID: {conv[1]}")
    print(f"  Title: {conv[2]}")
    print(f"  Source: {conv[3]}")

conn.close()
