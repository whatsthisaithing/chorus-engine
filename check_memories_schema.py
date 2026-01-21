"""Check memories table schema"""
from chorus_engine.db.database import engine
import sqlalchemy as sa

inspector = sa.inspect(engine)
tables = inspector.get_table_names()

print('Has memories table:', 'memories' in tables)

if 'memories' in tables:
    cols = inspector.get_columns('memories')
    print('\nColumns:')
    for c in cols:
        nullable = 'NULL' if c['nullable'] else 'NOT NULL'
        default = f" DEFAULT {c['default']}" if c.get('default') else ''
        print(f"  {c['name']}: {c['type']} {nullable}{default}")
