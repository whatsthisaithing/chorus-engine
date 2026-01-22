import sqlite3
import json

conn = sqlite3.connect('j:/Dev/chorus-engine/data/chorus.db')
cursor = conn.execute('''
    SELECT m.role, SUBSTR(m.content, 1, 60) as content, m.metadata 
    FROM messages m 
    JOIN threads t ON m.thread_id = t.id 
    WHERE t.conversation_id = ? 
    ORDER BY m.created_at 
    LIMIT 10
''', ('ad4ce9f0-fad4-4d5f-bfc5-1eb9321caed2',))

print('\nMessages with username metadata:')
print('=' * 80)
for i, (role, content, metadata) in enumerate(cursor, 1):
    username = 'N/A'
    if metadata:
        try:
            meta_dict = json.loads(metadata)
            username = meta_dict.get('username', 'N/A')
        except:
            pass
    print(f'{i}. {role:9s} ({username:20s}): {content}...')

conn.close()
