from typing import List, Set
from ...rag.models import Evidence, AnswerResponse, Chunk
import logging

logger = logging.getLogger(__name__)

def validate_citations_hard_fail(answer: AnswerResponse, used_chunks: List[Chunk], retry_fn=None) -> AnswerResponse:
    """
    Validates that every citation (evidence.chunk_id) in the answer exists in the used_chunks.
    If any citation is missing, abstain (return None) or retry once if retry_fn is provided.
    """
    chunk_ids: Set[str] = {chunk.id for chunk in used_chunks}
    missing = [ev.chunk_id for ev in answer.evidences if ev.chunk_id not in chunk_ids]
    if not missing:
        return answer
    logger.warning(f"Citations missing from used chunks: {missing}")
    if retry_fn is not None:
        logger.info("Retrying answer generation due to missing citations...")
        retry_answer = retry_fn()
        # Prevent infinite retry loop: only retry once
        missing_retry = [ev.chunk_id for ev in retry_answer.evidences if ev.chunk_id not in chunk_ids]
        if not missing_retry:
            return retry_answer
        logger.error(f"Citations still missing after retry: {missing_retry}. Abstaining.")
    return None  # Abstain if citations are invalid
