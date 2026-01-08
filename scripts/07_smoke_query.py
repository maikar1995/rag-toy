import asyncio
from rag_toy.api.deps import get_rag_service

q_1 = "Según el Exhibit 2, ¿qué razones se dan para afirmar que los cerdos son hermosos?"
q_2 = "¿Cuál es la opinión de Minto sobre el uso de tipografía Comic Sans en informes profesionales?"


async def main():
    svc = get_rag_service()
    res = await svc.ask(q_1)
    print(res)

if __name__ == "__main__":
    asyncio.run(main())