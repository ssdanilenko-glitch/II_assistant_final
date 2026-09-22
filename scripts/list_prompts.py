# scripts/list_prompts.py
"""Разовый список системных промптов из БД (без FastAPI lifespan)."""

import asyncio

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.chat.repositories.pg_models import SystemPromptRow
from app.core.config import get_settings


async def main() -> None:
    settings = get_settings()

    # ← подставь реальное поле, которое найдёшь командой выше
    dsn = settings.database_url

    engine = create_async_engine(dsn)
    sf = async_sessionmaker(engine, expire_on_commit=False)

    try:
        async with sf() as session:
            rows = (await session.execute(
                select(SystemPromptRow).order_by(SystemPromptRow.version)
            )).scalars().all()

            if not rows:
                print("system_prompts: пусто")
                return

            print(f"{'version':12s} {'pct':>4s} {'active':>7s} {'len':>6s}  preview")
            for r in rows:
                body = r.body or ""
                print(
                    f"{r.version:12s} {r.traffic_pct:>4d} "
                    f"{str(r.active):>7s} {len(body):>6d}  {body[:60]!r}"
                )
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())