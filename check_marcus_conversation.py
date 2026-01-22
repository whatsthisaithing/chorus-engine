import sqlite3
import json

# Query Marcus's conversation
conn = sqlite3.connect('j:/Dev/chorus-engine/data/chorus.db')

# First get the thread_id from the conversation
cursor = conn.execute('''
    SELECT id FROM threads 
    WHERE conversation_id = "7b03224a-11b1-4eb2-85f2-b3f6ec891ceb"
''')
threads = cursor.fetchall()
print(f"Found {len(threads)} thread(s) for this conversation")
if threads:
    thread_id = threads[0][0]
    print(f"Thread ID: {thread_id}")
    print()
else:
    print("No threads found!")
    exit()

cursor = conn.execute('''
    SELECT id, role, content, created_at, metadata 
    FROM messages 
    WHERE thread_id = ? 
    ORDER BY created_at
''', (thread_id,))

print("=" * 120)
print("MARCUS'S CONVERSATION: 7b03224a-11b1-4eb2-85f2-b3f6ec891ceb")
print("=" * 120)
print()

messages = cursor.fetchall()
for i, row in enumerate(messages, 1):
    msg_id, role, content, created_at, metadata = row
    
    print(f"Message {i}:")
    print(f"  ID: {msg_id}")
    print(f"  Role: {role}")
    print(f"  Content: {content[:100]}...")
    
    if metadata:
        try:
            metadata_dict = json.loads(metadata)
            print(f"  Metadata:")
            for key, value in metadata_dict.items():
                print(f"    {key}: {value}")
        except:
            print(f"  Metadata (raw): {metadata[:100]}...")
    
    print()

print(f"Total messages: {len(messages)}")
print()

# Check conversation details
cursor = conn.execute('''
    SELECT id, character_id, title, source 
    FROM conversations 
    WHERE id = "7b03224a-11b1-4eb2-85f2-b3f6ec891ceb"
''')
conv = cursor.fetchone()
if conv:
    print(f"Conversation Details:")
    print(f"  ID: {conv[0]}")
    print(f"  Character ID: {conv[1]}")
    print(f"  Title: {conv[2]}")
    print(f"  Source: {conv[3]}")

conn.close()
