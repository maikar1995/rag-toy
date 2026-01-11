"""
RAG Answer Generation Module

Generates answers with citations, confidence scoring, and abstention logic.
Uses Azure OpenAI for text generation with structured context and strict citation requirements.
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from openai import AzureOpenAI
from dotenv import load_dotenv
import os

# Load environment variables  
load_dotenv()

# Import SearchResult from retrieval module
from ..retrieval.retrieve import SearchResult

from ...rag.models import Citation, AnswerResponse, AbstractionReason

class AnswerGenerator:
    """
    RAG Answer Generator with citations, confidence scoring, and abstention.
    
    Features:
    - Structured context building with chunk IDs
    - Strict citation validation
    - Confidence scoring based on multiple factors
    - Abstention logic for insufficient evidence
    """
    
    def __init__(
        self,
        openai_client: Optional[AzureOpenAI] = None,
        model_deployment: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000
    ):
        """
        Initialize the answer generator.
        
        Args:
            openai_client: Azure OpenAI client (will create if None)
            model_deployment: Model deployment name (will load from env if None) 
            temperature: Generation temperature (0.1 for consistency)
            max_tokens: Maximum tokens in response
        """
        self.openai_client = openai_client or self._create_openai_client()
        self.model_deployment = model_deployment or self._get_model_deployment()
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        logging.info(f"✅ AnswerGenerator initialized: model={self.model_deployment}, temp={temperature}")
    
    def _create_openai_client(self) -> AzureOpenAI:
        """Create Azure OpenAI client for generation."""
        # Use same config as embeddings but potentially different deployment
        endpoint = os.getenv('GENERATION_1_ENDPOINT') or os.getenv('EMBEDDING_1_ENDPOINT')
        api_key = os.getenv('GENERATION_1_API_KEY') or os.getenv('EMBEDDING_1_API_KEY')
        api_version = os.getenv('GENERATION_1_API_VERSION') or os.getenv('EMBEDDING_1_API_VERSION', '2024-02-01')
        
        if not endpoint or not api_key:
            raise ValueError("Missing generation model credentials in environment")
        
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version
        )
    
    def _get_model_deployment(self) -> str:
        """Get model deployment name from environment."""
        deployment = (
            os.getenv('GENERATION_1_DEPLOYMENT') or 
            os.getenv('GENERATION_1_MODEL_NAME') or
            'gpt-4'  # Default fallback
        )
        return deployment
    
    def _build_structured_context(self, results: List[SearchResult]) -> str:
        """
        Build structured context from search results.
        
        Args:
            results: List of SearchResult objects
            
        Returns:
            Structured context string with chunk IDs
        """
        if not results:
            return "No relevant information found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            chunk_section = f"""[CHUNK {i}]
chunk_id: {result.chunk_id}
doc_id: {result.doc_id}
page: {result.page}
content: \"\"\"{result.content}\"\"\"

"""
            context_parts.append(chunk_section)
        
        return "\n".join(context_parts)
    
    def _create_generation_prompt(self, query: str, context: str) -> str:
        """
        Create prompt for answer generation.
        
        Args:
            query: User query
            context: Structured context from search results
            
        Returns:
            Complete prompt for the LLM
        """
        prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context.

STRICT RULES:
1. Use ONLY the chunks provided in the context below
2. Each important claim must be supported by citing the chunk_id 
3. If you cannot find sufficient evidence in the chunks, respond with "ABSTAIN"
4. Do not use any knowledge outside the provided context
5. Be precise and factual

RESPONSE FORMAT:
Provide your response as valid JSON with this exact structure:
{{
    "answer": "Your answer here or null if abstaining",
    "cited_chunks": ["chunk_id_1", "chunk_id_2"],
    "confidence_self_assessment": "high|medium|low",
    "reasoning": "Brief explanation of your confidence level"
}}

USER QUESTION: {query}

CONTEXT:
{context}

RESPONSE (valid JSON only):"""
        
        return prompt
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM response and extract structured data.
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            Parsed response dictionary
        """
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
                
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse LLM JSON response: {e}")
            logging.debug(f"Raw response: {response_text}")
            raise ValueError(f"Invalid JSON in LLM response: {e}")
    
    def _validate_citations(
        self, 
        cited_chunks: List[str], 
        available_chunks: List[SearchResult]
    ) -> List[Citation]:
        """
        Validate cited chunk IDs against available chunks.
        
        Args:
            cited_chunks: List of chunk IDs from LLM
            available_chunks: Available search results
            
        Returns:
            List of valid Citation objects
        """
        chunk_map = {result.chunk_id: result for result in available_chunks}
        valid_citations = []
        
        for chunk_id in cited_chunks:
            if chunk_id in chunk_map:
                result = chunk_map[chunk_id]
                citation = Citation(
                    chunk_id=chunk_id,
                    doc_id=result.doc_id,
                    page=result.page,
                    relevance=result.score
                )
                valid_citations.append(citation)
            else:
                logging.warning(f"LLM cited non-existent chunk: {chunk_id}")
        
        return valid_citations
    
    def _calculate_confidence(
        self,
        citations: List[Citation],
        search_results: List[SearchResult],
        self_assessment: str,
        search_type: str
    ) -> float:
        """
        Calculate confidence score based on multiple factors.
        
        Args:
            citations: Valid citations from LLM
            search_results: Original search results
            self_assessment: LLM's confidence self-assessment
            search_type: Type of search performed
            
        Returns:
            Confidence score between 0 and 1
        """
        confidence = 0.0
        
        # Base confidence from citations count
        if len(citations) >= 2:
            confidence += 0.2
        elif len(citations) >= 1:
            confidence += 0.1
        
        # Boost from high retrieval scores
        if citations:
            avg_relevance = sum(c.relevance for c in citations) / len(citations)
            if avg_relevance > 0.8:
                confidence += 0.2
            elif avg_relevance > 0.6:
                confidence += 0.1
        
        # Coherence bonus: citations from same document/page
        if len(citations) > 1:
            doc_ids = {c.doc_id for c in citations}
            pages = {(c.doc_id, c.page) for c in citations if c.page is not None}
            
            if len(doc_ids) == 1:  # Same document
                confidence += 0.1
            if len(pages) == 1:    # Same page
                confidence += 0.1
        
        # Self-assessment modifier
        if self_assessment == "high":
            confidence += 0.2
        elif self_assessment == "medium":
            confidence += 0.1
        elif self_assessment == "low":
            confidence -= 0.3
        
        # Search type bonus
        if search_type == "hybrid":
            confidence += 0.1
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))
    
    def _should_abstain(
        self, 
        llm_response: Dict[str, Any],
        citations: List[Citation],
        search_results: List[SearchResult]
    ) -> Optional[AbstractionReason]:
        """
        Determine if we should abstain from answering.
        
        Args:
            llm_response: Parsed LLM response
            citations: Validated citations
            search_results: Original search results
            
        Returns:
            AbstractionReason if should abstain, None otherwise
        """
        # No search results provided
        if not search_results:
            return AbstractionReason.NO_CHUNKS
        
        # LLM explicitly abstained
        if (llm_response.get("answer") is None or 
            llm_response.get("answer", "").upper().strip() == "ABSTAIN"):
            return AbstractionReason.INSUFFICIENT_EVIDENCE
        
        # No valid citations
        if not citations:
            return AbstractionReason.NO_CITATIONS
        
        # Very low confidence in self-assessment
        if llm_response.get("confidence_self_assessment") == "low":
            return AbstractionReason.INSUFFICIENT_EVIDENCE
        
        return None
    
    def generate(
        self, 
        query: str, 
        search_results: List[SearchResult],
        search_type: str = "hybrid"
    ) -> AnswerResponse:
        """
        Generate answer with citations and confidence scoring.
        
        Args:
            query: User query string
            search_results: List of SearchResult objects from retrieval
            search_type: Type of search performed (for confidence calculation)
            
        Returns:
            AnswerResponse object with answer, citations, confidence, and notes
        """
        logging.info(f"🤖 Generating answer for query: '{query[:50]}...'")
        logging.debug(f"Using {len(search_results)} search results")
        
        try:
            # Build structured context
            context = self._build_structured_context(search_results)
            
            # Create prompt
            prompt = self._create_generation_prompt(query, context)
            
            # Generate response
            response = self.openai_client.chat.completions.create(
                model=self.model_deployment,
                messages=[{"role": "user", "content": prompt}],
            )
            
            response_text = response.choices[0].message.content
            logging.debug(f"Raw LLM response: {response_text}")
            
            # Parse LLM response
            llm_response = self._parse_llm_response(response_text)
            
            # Validate citations
            cited_chunk_ids = llm_response.get("cited_chunks", [])
            valid_citations = self._validate_citations(cited_chunk_ids, search_results)
            
            # Check for abstention
            abstain_reason = self._should_abstain(llm_response, valid_citations, search_results)
            
            if abstain_reason:
                logging.info(f"🚫 Abstaining: {abstain_reason.value}")
                return AnswerResponse(
                    answer=None,
                    citations=[],
                    confidence=0.0,
                    notes={
                        "reason": abstain_reason.value,
                        "sources_count": len(search_results),
                        "search_type": search_type,
                        "model_used": self.model_deployment
                    }
                )
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                valid_citations,
                search_results,
                llm_response.get("confidence_self_assessment", "medium"),
                search_type
            )
            
            # Create successful response
            answer_response = AnswerResponse(
                answer=llm_response["answer"],
                citations=valid_citations,
                confidence=confidence,
                notes={
                    "sources_count": len(search_results),
                    "search_type": search_type,
                    "model_used": self.model_deployment,
                    "reasoning": llm_response.get("reasoning", ""),
                    "self_assessment": llm_response.get("confidence_self_assessment", "medium")
                }
            )
            
            logging.info(f"✅ Generated answer with {len(valid_citations)} citations, confidence: {confidence:.2f}")
            return answer_response
            
        except Exception as e:
            logging.error(f"❌ Answer generation failed: {e}")
            return AnswerResponse(
                answer=None,
                citations=[],
                confidence=0.0,
                notes={
                    "reason": AbstractionReason.GENERATION_ERROR.value,
                    "error": str(e),
                    "sources_count": len(search_results),
                    "search_type": search_type,
                    "model_used": self.model_deployment
                }
            )


def create_answer_generator(
    model_deployment: Optional[str] = None,
    temperature: float = 0.1
) -> AnswerGenerator:
    """
    Factory function to create an AnswerGenerator with default configuration.
    
    Args:
        model_deployment: Model deployment name (loads from env if None)
        temperature: Generation temperature
        
    Returns:
        Configured AnswerGenerator instance
    """
    return AnswerGenerator(
        model_deployment=model_deployment,
        temperature=temperature
    )


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    from ..retrieval.retrieve import create_retriever
    
    # Create components
    retriever = create_retriever()
    generator = create_answer_generator()
    
    # Example query
    query = "What is the Minto Pyramid Principle?"
    
    # Get search results
    search_results = retriever.search(query, top_k=5)
    
    # Generate answer
    response = generator.generate(query, search_results)
    
    # Print result
    print(json.dumps(response.to_dict(), indent=2, ensure_ascii=False))