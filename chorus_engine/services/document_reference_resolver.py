"""Document reference resolver for #doc:filename notation in conversations."""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session

from chorus_engine.repositories.document_repository import DocumentRepository

logger = logging.getLogger(__name__)


class DocumentReference:
    """Represents a document reference in user message."""
    
    def __init__(
        self,
        raw_text: str,
        filename: str,
        page_number: Optional[int] = None,
        document_id: Optional[int] = None
    ):
        """
        Initialize document reference.
        
        Args:
            raw_text: Original reference text (e.g., "#doc:report.pdf#page-5")
            filename: Document filename
            page_number: Optional page number
            document_id: Resolved document ID (if found)
        """
        self.raw_text = raw_text
        self.filename = filename
        self.page_number = page_number
        self.document_id = document_id
    
    @property
    def is_resolved(self) -> bool:
        """Check if document reference has been resolved."""
        return self.document_id is not None
    
    def __repr__(self):
        return f"<DocumentReference('{self.filename}', page={self.page_number}, resolved={self.is_resolved})>"


class DocumentReferenceResolver:
    """Service for resolving document references in user messages."""
    
    # Pattern: #doc:filename.ext or #doc:filename.ext#page-5
    DOC_REFERENCE_PATTERN = re.compile(
        r'#doc:([a-zA-Z0-9_\-\.]+?)(?:#page-(\d+))?(?:\s|$|[,\.!?])',
        re.IGNORECASE
    )
    
    def __init__(self):
        """Initialize document reference resolver."""
        logger.info("DocumentReferenceResolver initialized")
    
    def find_references(self, text: str) -> List[DocumentReference]:
        """
        Find all document references in text.
        
        Args:
            text: User message text
            
        Returns:
            List of DocumentReference objects (unresolved)
        """
        references = []
        
        for match in self.DOC_REFERENCE_PATTERN.finditer(text):
            raw_text = match.group(0).strip()
            filename = match.group(1)
            page_str = match.group(2)
            page_number = int(page_str) if page_str else None
            
            ref = DocumentReference(
                raw_text=raw_text,
                filename=filename,
                page_number=page_number
            )
            references.append(ref)
        
        return references
    
    def resolve_references(
        self,
        text: str,
        db: Session,
        character_id: Optional[str] = None
    ) -> Tuple[List[DocumentReference], str]:
        """
        Find and resolve document references in text.
        
        Args:
            text: User message text
            db: Database session
            character_id: Character ID for scoped document access
            
        Returns:
            Tuple of (resolved_references, cleaned_text)
            - resolved_references: List with document_ids populated
            - cleaned_text: Text with references replaced by readable format
        """
        references = self.find_references(text)
        
        if not references:
            return [], text
        
        repo = DocumentRepository(db)
        cleaned_text = text
        
        # Resolve each reference
        for ref in references:
            # Try exact filename match with character filtering
            documents = repo.list_documents(
                limit=1000,
                character_id=character_id,
                include_global=True
            )
            matched_doc = None
            
            for doc in documents:
                if doc.filename.lower() == ref.filename.lower():
                    matched_doc = doc
                    break
                # Also try matching without extension
                if Path(doc.filename).stem.lower() == Path(ref.filename).stem.lower():
                    matched_doc = doc
                    break
            
            if matched_doc:
                ref.document_id = matched_doc.id
                
                # Replace reference with readable format
                if ref.page_number:
                    readable = f"[Document: {matched_doc.title}, page {ref.page_number}]"
                else:
                    readable = f"[Document: {matched_doc.title}]"
                
                cleaned_text = cleaned_text.replace(ref.raw_text, readable + " ")
                
                logger.debug(f"Resolved reference: {ref.filename} -> document {matched_doc.id}")
            else:
                logger.warning(f"Could not resolve document reference: {ref.filename}")
                # Leave unresolved references as-is but make them readable
                cleaned_text = cleaned_text.replace(ref.raw_text, f"[Document: {ref.filename} (not found)] ")
        
        resolved_refs = [ref for ref in references if ref.is_resolved]
        
        return resolved_refs, cleaned_text
    
    def get_autocomplete_suggestions(
        self,
        partial_filename: str,
        db: Session,
        character_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get autocomplete suggestions for document references.
        
        Args:
            partial_filename: Partial filename being typed
            db: Database session
            character_id: Character ID for scoped document access
            limit: Maximum suggestions to return
            
        Returns:
            List of suggestion dictionaries
        """
        repo = DocumentRepository(db)
        documents = repo.list_documents(
            limit=1000,
            status="completed",
            character_id=character_id,
            include_global=True
        )
        
        partial_lower = partial_filename.lower()
        suggestions = []
        
        for doc in documents:
            filename_lower = doc.filename.lower()
            title_lower = doc.title.lower() if doc.title else ""
            
            # Match on filename or title
            if partial_lower in filename_lower or partial_lower in title_lower:
                suggestions.append({
                    "document_id": doc.id,
                    "filename": doc.filename,
                    "title": doc.title,
                    "file_type": doc.file_type,
                    "reference_syntax": f"#doc:{doc.filename}",
                    "has_pages": doc.page_count is not None,
                    "page_count": doc.page_count
                })
                
                if len(suggestions) >= limit:
                    break
        
        return suggestions


from pathlib import Path
