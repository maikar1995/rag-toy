import asyncio
from rag_toy.api.deps import get_rag_service

async def main():
    svc = get_rag_service()
    res = await svc.ask("Según el Exhibit 2, ¿qué razones se dan para afirmar que los cerdos son hermosos?")
    print(res)

if __name__ == "__main__":
    asyncio.run(main())