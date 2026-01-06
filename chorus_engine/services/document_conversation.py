"""Document-aware conversation service with automatic chunk retrieval and citation."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session

from chorus_engine.services.document_reference_resolver import DocumentReferenceResolver
from chorus_engine.services.document_vector_store import DocumentVectorStore
from chorus_engine.repositories.document_repository import DocumentRepository

logger = logging.getLogger(__name__)


class DocumentContext:
    """Container for document chunks injected into conversation."""
    
    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.citations: List[str] = []
    
    def add_chunk(
        self,
        content: str,
        document_id: int,
        document_title: str,
        chunk_index: int,
        page_numbers: Optional[str] = None,
        relevance_score: float = 0.0
    ):
        """Add a chunk to the context."""
        self.chunks.append({
            "content": content,
            "document_id": document_id,
            "document_title": document_title,
            "chunk_index": chunk_index,
            "page_numbers": page_numbers,
            "relevance_score": relevance_score
        })
        
        # Build citation
        citation = f"{document_title}"
        if page_numbers:
            citation += f", pages {page_numbers}"
        
        if citation not in self.citations:
            self.citations.append(citation)
    
    def format_context_injection(self) -> str:
        """Format chunks for injection into LLM context."""
        if not self.chunks:
            return ""
        
        lines = [
            "",
            "=== RELEVANT DOCUMENT EXCERPTS ===",
            "",
            "The following information from uploaded documents may be relevant to this conversation:",
            ""
        ]
        
        for i, chunk in enumerate(self.chunks, 1):
            lines.append(f"[Excerpt {i} from {chunk['document_title']}]")
            if chunk['page_numbers']:
                lines.append(f"(Pages: {chunk['page_numbers']})")
            lines.append(chunk['content'])
            lines.append("")
        
        lines.append("=== END DOCUMENT EXCERPTS ===")
        lines.append("")
        
        return "\n".join(lines)
    
    def format_citations(self) -> str:
        """Format citations to append to assistant response."""
        if not self.citations:
            return ""
        
        if len(self.citations) == 1:
            return f"\n\n*Source: {self.citations[0]}*"
        else:
            citation_list = "\n".join(f"• {cite}" for cite in self.citations)
            return f"\n\n*Sources:*\n{citation_list}"
    
    def has_content(self) -> bool:
        """Check if any chunks were found."""
        return len(self.chunks) > 0


class DocumentConversationService:
    """Service for integrating documents into conversations."""
    
    def __init__(
        self,
        vector_store: DocumentVectorStore,
        reference_resolver: DocumentReferenceResolver
    ):
        """
        Initialize document conversation service.
        
        Args:
            vector_store: Document vector store
            reference_resolver: Reference resolver
        """
        self.vector_store = vector_store
        self.reference_resolver = reference_resolver
        
        logger.info("DocumentConversationService initialized")
    
    def process_user_message(
        self,
        user_message: str,
        db: Session,
        conversation_id: Optional[str] = None,
        character_id: Optional[str] = None,
        n_results: int = 3
    ) -> Tuple[str, DocumentContext]:
        """
        Process user message to detect and resolve document needs.
        
        This method:
        1. Detects explicit document references (#doc:filename)
        2. Performs semantic search if message is a question
        3. Retrieves relevant chunks
        4. Returns cleaned message and document context
        
        Args:
            user_message: Original user message
            db: Database session
            conversation_id: Optional conversation ID for logging
            character_id: Character ID for scoped document access
            n_results: Number of chunks to retrieve per query
            
        Returns:
            Tuple of (processed_message, document_context)
        """
        doc_context = DocumentContext()
        repo = DocumentRepository(db)
        
        # Step 1: Check for explicit document references
        resolved_refs, cleaned_message = self.reference_resolver.resolve_references(
            user_message,
            db,
            character_id=character_id
        )
        
        if resolved_refs:
            logger.info(f"Found {len(resolved_refs)} document references")
            
            # Retrieve chunks from referenced documents
            for ref in resolved_refs:
                # Verify document access (Phase 9 - scope validation)
                if character_id and not repo.verify_document_access(
                    document_id=ref.document_id,
                    character_id=character_id,
                    conversation_id=conversation_id
                ):
                    logger.warning(
                        f"Document {ref.document_id} not accessible in current context "
                        f"(character={character_id}, conversation={conversation_id})"
                    )
                    continue
                
                chunks = self._retrieve_from_document(
                    document_id=ref.document_id,
                    query=user_message,
                    page_number=ref.page_number,
                    n_results=n_results
                )
                
                logger.debug(f"Retrieved {len(chunks)} chunks from document {ref.document_id}")
                
                for chunk in chunks:
                    logger.debug(f"Adding chunk with relevance {chunk.get('relevance_score', 0):.2f}")
                    doc_context.add_chunk(**chunk)
                
                # Log access
                if conversation_id:
                    repo.log_document_access(
                        document_id=ref.document_id,
                        access_type="reference",
                        conversation_id=conversation_id,
                        chunks_retrieved=len(chunks),
                        query=user_message
                    )
        
        # Step 2: Check if message is a question (semantic search)
        elif self._is_question(user_message):
            logger.debug("Detected question - performing semantic document search")
            
            # Get accessible document IDs first (scope-aware)
            accessible_doc_ids = None
            if character_id:
                accessible_docs = repo.get_accessible_documents(
                    character_id=character_id,
                    conversation_id=conversation_id,
                    include_character_scope=True,
                    include_global=True,
                    limit=1000  # Get all accessible documents
                )
                accessible_doc_ids = [doc.id for doc in accessible_docs]
                logger.debug(f"Scope filtering: {len(accessible_doc_ids)} accessible documents")
            
            # Search documents with scope-aware filtering (Phase 9)
            chunk_ids, texts, metadatas, distances = self.vector_store.search_with_scope(
                query=user_message,
                character_id=character_id or "",
                conversation_id=conversation_id,
                n_results=n_results,
                document_ids=accessible_doc_ids
            )
            
            # Add relevant chunks (filter by relevance threshold)
            for i in range(len(chunk_ids)):
                relevance_score = 1.0 - distances[i]
                
                # Only include chunks with reasonable relevance (>0.5)
                if relevance_score > 0.5:
                    doc_context.add_chunk(
                        content=texts[i],
                        document_id=metadatas[i].get("document_id"),
                        document_title=metadatas[i].get("document_title", "Unknown"),
                        chunk_index=metadatas[i].get("chunk_index", 0),
                        page_numbers=metadatas[i].get("page_numbers"),
                        relevance_score=relevance_score
                    )
                    
                    # Log access
                    if conversation_id:
                        repo.log_document_access(
                            document_id=metadatas[i].get("document_id"),
                            access_type="retrieval",
                            conversation_id=conversation_id,
                            chunks_retrieved=1,
                            query=user_message,
                            relevance_score=relevance_score
                        )
        
        logger.debug(f"Returning doc_context with {len(doc_context.chunks)} chunks")
        return cleaned_message, doc_context
    
    def _retrieve_from_document(
        self,
        document_id: int,
        query: str,
        page_number: Optional[int] = None,
        n_results: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks from specific document.
        
        Args:
            document_id: Document ID
            query: Search query
            page_number: Optional page filter
            n_results: Number of chunks
            
        Returns:
            List of chunk dictionaries
        """
        # Search within document
        chunk_ids, texts, metadatas, distances = self.vector_store.search(
            query=query,
            n_results=n_results * 3,  # Get more for filtering
            document_id=document_id
        )
        
        chunks = []
        
        for i in range(len(chunk_ids)):
            # Filter by page if specified
            if page_number:
                page_nums = metadatas[i].get("page_numbers")
                if page_nums:
                    # Parse page range (e.g., "5-7" or "12")
                    if "-" in str(page_nums):
                        start, end = map(int, str(page_nums).split("-"))
                        if not (start <= page_number <= end):
                            continue
                    elif int(page_nums) != page_number:
                        continue
            
            chunks.append({
                "content": texts[i],
                "document_id": metadatas[i].get("document_id"),
                "document_title": metadatas[i].get("document_title", "Unknown"),
                "chunk_index": metadatas[i].get("chunk_index", 0),
                "page_numbers": metadatas[i].get("page_numbers"),
                "relevance_score": 1.0 - distances[i]
            })
            
            if len(chunks) >= n_results:
                break
        
        return chunks
    
    def _is_question(self, text: str) -> bool:
        """
        Detect if text is a question.
        
        Args:
            text: User message
            
        Returns:
            True if appears to be a question
        """
        text_lower = text.lower().strip()
        
        # Check for question mark
        if "?" in text:
            return True
        
        # Check for question words at start
        question_words = [
            "what", "when", "where", "who", "why", "how",
            "is", "are", "can", "could", "would", "should",
            "does", "do", "did", "has", "have", "will"
        ]
        
        first_word = text_lower.split()[0] if text_lower else ""
        return first_word in question_words
    
    def inject_context_into_prompt(
        self,
        prompt: str,
        doc_context: DocumentContext,
        position: str = "after_system"
    ) -> str:
        """
        Inject document context into prompt.
        
        Args:
            prompt: Original prompt
            doc_context: Document context to inject
            position: Where to inject ("after_system" or "before_user")
            
        Returns:
            Modified prompt with document context
        """
        if not doc_context.has_content():
            return prompt
        
        context_text = doc_context.format_context_injection()
        
        if position == "after_system":
            # Find end of system message (look for first user message)
            parts = prompt.split("\n\n")
            if len(parts) > 1:
                return parts[0] + "\n\n" + context_text + "\n\n" + "\n\n".join(parts[1:])
            else:
                return prompt + "\n\n" + context_text
        else:  # before_user
            return prompt + "\n\n" + context_text
    
    def append_citations_to_response(
        self,
        response: str,
        doc_context: DocumentContext
    ) -> str:
        """
        Append citations to assistant response.
        
        Args:
            response: Original assistant response
            doc_context: Document context used
            
        Returns:
            Response with citations appended
        """
        if not doc_context.has_content():
            return response
        
        citations = doc_context.format_citations()
        return response + citations
