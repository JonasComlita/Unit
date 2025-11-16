import asyncio
import os
import asyncpg

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))


async def main():
    # Attempt to load a .env file from the project root (parent of scripts/)
    # This is a tiny, dependency-free loader so scripts run correctly even
    # when you don't `export` env vars into the shell.
    def _load_dotenv(env_path):
        if not os.path.exists(env_path):
            return
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' not in line:
                    continue
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.strip()
                # Remove surrounding quotes if present
                if (val.startswith('"') and val.endswith('"')) or (
                    val.startswith("'") and val.endswith("'")
                ):
                    val = val[1:-1]
                # Only set if not already in environment
                os.environ.setdefault(key, val)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dotenv_path = os.path.join(project_root, '.env')
    _load_dotenv(dotenv_path)

    db_url = os.getenv('DATABASE_URL')
    pool = await asyncpg.create_pool(db_url, min_size=1, max_size=3)
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT 1 as ok")
        print("DB test row:", row)
    await pool.close()

if __name__ == "__main__":
    asyncio.run(main())