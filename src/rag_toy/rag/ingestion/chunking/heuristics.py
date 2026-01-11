"""Heuristics for intelligent chunking decisions."""

import re
from typing import Dict, Any
from ...models import Document


def is_slide_like(doc: Document) -> bool:
    """
    Determine if a document is slide-like (presentations, brief content).
    
    Heuristics:
    - Short content overall
    - Many short lines  
    - High line break ratio
    
    Args:
        doc: Document to analyze
        
    Returns:
        True if document appears to be slide-like
    """
    content = getattr(doc, 'content', None)
    if not content:
        return False
        
    content_length = len(content)
    
    # Very short content is likely slide-like
    if content_length < 500:
        return True
        
    # Count lines and calculate metrics
    lines = content.split('\n')
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    
    if not non_empty_lines:
        return False
        
    avg_line_length = sum(len(line) for line in non_empty_lines) / len(non_empty_lines)
    line_break_ratio = content.count('\n') / content_length
    
    # Slide heuristics
    is_short_lines = avg_line_length < 50  # Short average line length
    is_high_break_ratio = line_break_ratio > 0.05  # High proportion of line breaks
    is_moderate_length = 500 < content_length < 3000  # Moderate content length
    
    return is_short_lines and is_high_break_ratio and is_moderate_length


def needs_recursive_chunking(doc: Document) -> bool:
    """
    Determine if a document needs recursive chunking.
    
    Heuristics:
    - Very long content that exceeds typical chunk sizes
    
    Args:
        doc: Document to analyze
        
    Returns:
        True if document should use recursive chunking
    """
    content = getattr(doc, 'content', None)
    if not content:
        return False
        
    # Long documents benefit from recursive chunking
    content_length = len(content)
    return content_length > 5000


def default_chunk_params(doc: Document) -> Dict[str, Any]:
    """
    Generate default chunking parameters based on document characteristics.
    
    Args:
        doc: Document to analyze
        
    Returns:
        Dictionary with recommended chunking parameters
    """
    if is_slide_like(doc):
        # Slides: smaller chunks, no overlap to preserve slide boundaries
        return {
            "min_chars": 50,
            "max_chars": 800,
            "overlap_chars": 0,
        }
    elif needs_recursive_chunking(doc):
        # Long documents: larger chunks with moderate overlap
        return {
            "min_chars": 200,
            "max_chars": 1500,
            "overlap_chars": 150,
        }
    else:
        # Standard text: balanced approach
        return {
            "min_chars": 100,
            "max_chars": 1200,
            "overlap_chars": 100,
        }