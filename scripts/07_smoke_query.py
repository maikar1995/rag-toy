import asyncio
from rag_toy.api.deps import get_rag_service

q_1 = "Según el Exhibit 2, ¿qué razones se dan para afirmar que los cerdos son hermosos?"
q_2 = "¿Cuál es la opinión de Minto sobre el uso de tipografía Comic Sans en informes profesionales?"
q_3 = "Explícame la diferencia entre deductive e inductive grouping en la Pyramid Principle, y dame una regla para saber cuál usar."
q_4 = "Enumera los pasos para construir una pirámide desde el mensaje principal hasta los supporting points, en orden, y explica cada paso con una frase."
q_5 = "Resume las reglas MECE que menciona el libro y dame un ejemplo de mala agrupación y cómo corregirla."
q_6 = "¿Qué justificación da el libro para empezar por el key message antes que por los detalles? Responde con 2 razones y apóyalas con el texto."


async def main():
    svc = get_rag_service()
    res = await svc.ask(q_6)
    print(res)

if __name__ == "__main__":
    asyncio.run(main())