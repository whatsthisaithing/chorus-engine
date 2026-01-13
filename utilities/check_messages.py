"""Quick script to check message history in a thread."""
import sqlite3
import sys

thread_id = 'bcc03760-889e-4327-a260-add80c752975'  # Replace with your thread ID

conn = sqlite3.connect('data/chorus.db')
cursor = conn.cursor()

print(f"\nLast 5 messages in thread {thread_id}:\n")
cursor.execute('''
    SELECT id, role, content, created_at 
    FROM messages 
    WHERE thread_id=? 
    ORDER BY created_at DESC 
    LIMIT 5
''', (thread_id,))

for i, row in enumerate(reversed(cursor.fetchall())):
    msg_id, role, content, created = row
    print(f"{'='*80}")
    print(f"Message {i+1}: {msg_id[:8]}... | {role} | {created}")
    print(f"{'='*80}")
    print(content)
    print()

conn.close()
