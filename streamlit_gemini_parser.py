import streamlit as st
import json
import re
import pandas as pd
import subprocess
import sys
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Import NLP libraries with fallbacks
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    device = 'cpu'

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="🔍 Gemini Data Extractor",
    page_icon="🎯",
    layout="wide"
)

st.markdown("""
<style>
    /* Force dark theme and high contrast */
    .stApp {
        background-color: #0e1117 !important;
        color: #ffffff !important;
    }
    
    .main-header {
        font-size: 2.5rem;
        color: #00d4ff !important;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .extraction-section {
        background: #1a1a1a !important;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #00ff88;
        color: #ffffff !important;
    }
    
    .entity-item {
        background: #2d2d2d !important;
        padding: 0.8rem;
        margin: 0.4rem 0;
        border-radius: 0.5rem;
        border-left: 4px solid #00d4ff;
        color: #ffffff !important;
    }
    
    .nlp-entity {
        background: #1a3d1a !important;
        padding: 0.8rem;
        margin: 0.4rem 0;
        border-radius: 0.5rem;
        border-left: 4px solid #00ff88;
        color: #ffffff !important;
    }
    
    .query-item {
        background: #3d2a1a !important;
        padding: 0.8rem;
        margin: 0.4rem 0;
        border-radius: 0.5rem;
        border-left: 4px solid #ffaa00;
        color: #ffffff !important;
    }
    
    .snippet-preview {
        background: #1a1a1a !important;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #333333;
        color: #ffffff !important;
        font-family: 'Courier New', monospace;
        white-space: pre-wrap;
    }
    
    /* Override all Streamlit text colors */
    .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span {
        color: #ffffff !important;
    }
    
    .stExpander {
        background-color: #1a1a1a !important;
        border: 1px solid #333333 !important;
    }
    
    .stExpander > div > div {
        color: #ffffff !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #ffffff !important;
    }
    
    .stInfo {
        background-color: #1a3d3d !important;
        color: #ffffff !important;
    }
    
    .stSuccess {
        background-color: #1a3d1a !important;
        color: #ffffff !important;
    }
    
    .stWarning {
        background-color: #3d3d1a !important;
        color: #ffffff !important;
    }
    
    .stError {
        background-color: #3d1a1a !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SPACY MODEL DOWNLOAD
# ============================================================================

def download_spacy_model():
    """Download spaCy model if not available"""
    if not SPACY_AVAILABLE:
        return False
    
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        return True
    except OSError:
        try:
            st.info("📥 Downloading spaCy model... This may take a moment.")
            result = subprocess.run([
                sys.executable, "-m", "spacy", "download", "en_core_web_sm"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                st.success("✅ spaCy model downloaded!")
                nlp = spacy.load("en_core_web_sm")
                return True
            else:
                return False
        except Exception as e:
            st.warning(f"⚠️ Could not download spaCy model: {str(e)[:100]}...")
            return False

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ExtractedData:
    original_query: str
    suggested_queries: List[str]
    text_snippets: List[str]
    links: List[Dict[str, str]]
    entities: List[Dict[str, Any]]

# ============================================================================
# NLP ENTITY EXTRACTOR
# ============================================================================

@st.cache_resource
class NLPEntityExtractor:
    """Advanced NLP-based entity extraction"""
    
    def __init__(self):
        self.ner_pipeline = None
        self.nlp = None
        self.is_initialized = False
        self.model_type = None
        
        # Fallback patterns
        self.fallback_patterns = {
            'PERSON': r'\b(?:President|Prime Minister|CEO|Director|Mr\.|Mrs\.|Dr\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|Donald Trump|Joe Biden|Vladimir Putin|Xi Jinping|Sara Duterte|Ayatollah Ali Khamenei',
            'ORG': r'\b(?:UN|NATO|FIFA|WHO|Google|Apple|Microsoft|Amazon|McDonald\'s|Tesla|The People\'s Union USA|PDP-Laban|Oklahoma City Thunder|FC Porto|Al Ahly)\b',
            'LOC': r'\b(?:United States|US|USA|Iran|Israel|China|Russia|Middle East|Europe|Asia|Qatar|Philippines|Nigeria|Fordow|Natanz|Isfahan|MetLife Stadium)\b',
            'DATE': r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}|\b\d{1,2}/\d{1,2}/\d{4}\b|June 24, 2025',
            'MONEY': r'\$\d+(?:,\d{3})*(?:\.\d{2})?(?:\s*(?:million|billion|trillion))?',
            'EVENT': r'\bFIFA Club World Cup 2025|NBA championship|World Cup|Olympics\b'
        }
    
    def initialize(self, model_choice="auto"):
        """Initialize NLP models with fallback"""
        
        if self.is_initialized:
            return True
        
        try:
            # Try transformer models first
            if model_choice in ["bert-base", "auto"] and TRANSFORMERS_AVAILABLE:
                try:
                    with st.spinner("🧠 Loading BERT model for entity extraction..."):
                        self.ner_pipeline = pipeline(
                            "ner",
                            model="dslim/bert-base-NER",
                            aggregation_strategy="simple",
                            device=-1  # Force CPU for stability
                        )
                    self.model_type = "bert-base"
                    self.is_initialized = True
                    st.success("✅ BERT model loaded for entity extraction!")
                    return True
                except Exception as e:
                    st.warning(f"⚠️ BERT failed: {str(e)[:50]}... Trying spaCy.")
            
            # Try spaCy
            if SPACY_AVAILABLE and download_spacy_model():
                try:
                    import spacy
                    self.nlp = spacy.load("en_core_web_sm")
                    self.model_type = "spacy"
                    self.is_initialized = True
                    st.success("✅ spaCy model loaded for entity extraction!")
                    return True
                except Exception as e:
                    st.warning(f"⚠️ spaCy failed: {str(e)[:50]}... Using patterns.")
            
            # Fallback to patterns
            self.model_type = "patterns"
            self.is_initialized = True
            st.info("ℹ️ Using pattern matching for entity extraction.")
            return True
            
        except Exception as e:
            st.warning(f"NLP initialization failed: {str(e)[:50]}... Using basic patterns.")
            self.model_type = "patterns"
            self.is_initialized = True
            return True
    
    def extract_entities(self, text_snippets: List[str]) -> List[Dict[str, Any]]:
        """Extract entities using NLP models with fallback"""
        
        if not self.is_initialized:
            self.initialize()
        
        all_text = " ".join(text_snippets)
        entities = []
        
        try:
            # Try BERT transformer
            if self.ner_pipeline:
                results = self.ner_pipeline(all_text)
                for result in results:
                    entities.append({
                        'text': result["word"],
                        'type': result["entity_group"],
                        'confidence': result["score"],
                        'source': 'BERT',
                        'start': result["start"],
                        'end': result["end"]
                    })
            
            # Try spaCy
            elif self.nlp:
                doc = self.nlp(all_text)
                for ent in doc.ents:
                    entities.append({
                        'text': ent.text,
                        'type': ent.label_,
                        'confidence': 0.85,  # spaCy doesn't provide confidence
                        'source': 'spaCy',
                        'start': ent.start_char,
                        'end': ent.end_char
                    })
            
            # Fallback to patterns
            else:
                entities = self._extract_with_patterns(all_text)
            
            # Remove duplicates and add context
            return self._post_process_entities(entities, all_text)
            
        except Exception as e:
            st.warning(f"Entity extraction error: {str(e)[:50]}... Using fallback patterns.")
            return self._extract_with_patterns(all_text)
    
    def _extract_with_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Fallback pattern-based extraction"""
        entities = []
        
        for entity_type, pattern in self.fallback_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'text': match.group().strip(),
                    'type': entity_type,
                    'confidence': 0.75,
                    'source': 'patterns',
                    'start': match.start(),
                    'end': match.end()
                })
        
        return entities
    
    def _post_process_entities(self, entities: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        """Remove duplicates and add context"""
        if not entities:
            return []
        
        # Sort by position
        entities.sort(key=lambda x: x['start'])
        
        # Remove overlaps - keep higher confidence
        unique_entities = []
        for entity in entities:
            overlap = False
            for i, existing in enumerate(unique_entities):
                if (entity['start'] < existing['end'] and entity['end'] > existing['start']):
                    if entity['confidence'] > existing['confidence']:
                        unique_entities[i] = entity
                    overlap = True
                    break
            
            if not overlap:
                unique_entities.append(entity)
        
        # Add context for each entity
        for entity in unique_entities:
            start_ctx = max(0, entity['start'] - 30)
            end_ctx = min(len(text), entity['end'] + 30)
            entity['context'] = text[start_ctx:end_ctx].replace('\n', ' ').strip()
        
        return unique_entities

# ============================================================================
# GEMINI DATA EXTRACTOR
# ============================================================================

class GeminiDataExtractor:
    """Enhanced extractor with NLP entity extraction"""
    
    def __init__(self):
        self.nlp_extractor = NLPEntityExtractor()
    
    def extract_from_gemini_raw(self, raw_data: str, use_nlp: bool = True) -> ExtractedData:
        """Extract all requested data from raw Gemini response"""
        
        # Extract original query
        original_query = self._extract_original_query(raw_data)
        
        # Extract suggested queries
        suggested_queries = self._extract_suggested_queries(raw_data)
        
        # Extract text snippets
        text_snippets = self._extract_text_snippets(raw_data)
        
        # Extract links
        links = self._extract_links(raw_data)
        
        # Extract entities using NLP
        if use_nlp:
            if not self.nlp_extractor.is_initialized:
                self.nlp_extractor.initialize()
            entities = self.nlp_extractor.extract_entities(text_snippets)
        else:
            entities = self._extract_entities_basic(text_snippets)
        
        return ExtractedData(
            original_query=original_query,
            suggested_queries=suggested_queries,
            text_snippets=text_snippets,
            links=links,
            entities=entities
        )
    
    def _extract_entities_basic(self, text_snippets: List[str]) -> List[Dict[str, Any]]:
        """Basic pattern extraction for comparison"""
        entities = []
        all_text = " ".join(text_snippets)
        
        basic_patterns = {
            'PERSON': r'\b(?:President|Prime Minister)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|Donald Trump|Joe Biden',
            'ORG': r'\b(?:UN|NATO|FIFA|Google|Apple|Microsoft)\b',
            'LOC': r'\b(?:United States|US|Iran|Israel|China|Russia)\b'
        }
        
        for entity_type, pattern in basic_patterns.items():
            matches = re.finditer(pattern, all_text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'text': match.group().strip(),
                    'type': entity_type,
                    'confidence': 0.60,
                    'source': 'basic_patterns',
                    'start': match.start(),
                    'end': match.end(),
                    'context': ''
                })
        
        return entities
    
    def _extract_original_query(self, raw_data: str) -> str:
        """Extract the original search query with multiple patterns"""
        patterns = [
            r'\[\[\\?"([^"]+)\\?"\],3,null,0',     # Main query pattern with optional escaped quotes
            r'\[\[\\?"([^"]+)\\?"\]',              # Alternative pattern with optional escaped quotes
            r'"([^"]*latest news[^"]*)"',          # News-specific pattern
            r'null,\[\[\\?"([^"]+)\\?"\],3',       # Another common pattern with escaped quotes
            r'\[\\?"([^"]+)\\?",3,null,0',         # Variation without brackets with escaped quotes
            r'\[\["([^"]+)"\],3,null,0',           # Original pattern without escaped quotes
            r'\[\["([^"]+)"\]',                    # Original alternative pattern
            r'null,\[\["([^"]+)"\],3',             # Original common pattern
            r'\["([^"]+)",3,null,0',               # Original variation without brackets
        ]
        
        for pattern in patterns:
            match = re.search(pattern, raw_data, re.IGNORECASE)
            if match:
                query = match.group(1).strip()
                # Clean up any remaining escaped quotes
                query = query.replace('\\"', '"').replace('\\\\', '\\')
                if len(query) > 2:  # Valid query
                    return query
        
        return "Query not found"
    
    def _extract_suggested_queries(self, raw_data: str) -> List[str]:
        """Extract ALL suggested/related queries from Gemini response"""
        suggested = []
        
        # Multiple patterns for different query formats in Gemini
        patterns = [
            r'\[\\?"([^"]+)\\?",1\]',                    # Format: [["query",1]] or [\"query\",1]
            r'\[\\?"([^"]+)\\?",2\]',                    # Format: [["query",2]] or [\"query\",2]
            r'\[\\?"([^"]+)\\?",4\]',                    # Format: [["query",4]] or [\"query\",4]
            r'\[\["([^"]+)",1\]',                        # Original format: [["query",1]]
            r'\[\["([^"]+)",2\]',                        # Original format: [["query",2]]
            r'\[\["([^"]+)",4\]',                        # Original format: [["query",4]]
            r'"([^"]*(?:what|how|when|where|why|who)[^"]*)"',  # Question patterns
            r'"([^"]*(?:latest|breaking|news|today)[^"]*)"',   # News patterns
            r'"([^"]*\?[^"]*)"',                         # Any question with ?
            r'\[\\?"([^"]+)\\?",[1-9]\]',                # General pattern with numbers (escaped quotes)
            r'\["([^"]+)",[1-9]\]',                      # General pattern with numbers (regular quotes)
        ]
        
        all_matches = set()  # Use set to avoid duplicates
        
        for pattern in patterns:
            matches = re.findall(pattern, raw_data, re.IGNORECASE)
            for match in matches:
                cleaned_match = match.strip()
                # Clean up escaped quotes
                cleaned_match = cleaned_match.replace('\\"', '"').replace('\\\\', '\\')
                
                # Filter valid queries
                if (len(cleaned_match) > 5 and 
                    not cleaned_match.startswith('http') and
                    not cleaned_match.startswith('The latest news') and
                    cleaned_match not in all_matches):
                    all_matches.add(cleaned_match)
        
        # Convert to list and sort by relevance
        suggested = list(all_matches)
        
        # Prioritize question-style queries
        question_queries = [q for q in suggested if any(word in q.lower() for word in ['what', 'how', 'when', 'where', 'why', 'who', '?'])]
        news_queries = [q for q in suggested if any(word in q.lower() for word in ['news', 'latest', 'breaking', 'today', 'headlines'])]
        other_queries = [q for q in suggested if q not in question_queries and q not in news_queries]
        
        # Combine in priority order
        final_queries = question_queries + news_queries + other_queries
        
        return final_queries[:15]  # Return more queries
    
    def _extract_text_snippets(self, raw_data: str) -> List[str]:
        """Extract main text content/snippets"""
        snippets = []
        
        # Main text content patterns
        patterns = [
            r'"(The latest news[^"]*(?:\\"[^"]*)*)"',
            r'"(Breaking news[^"]*(?:\\"[^"]*)*)"',
            r'"([^"]{200,})"',  # Any substantial text
            r'"([^"]*(?:President|Iran|Israel|news|announced)[^"]*)"'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, raw_data, re.IGNORECASE)
            for match in matches:
                # Clean up the text
                cleaned = match.replace('\\"', '"').replace('\\n', ' ').replace('\\t', ' ')
                cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                
                # Only add substantial, unique snippets
                if len(cleaned) > 50 and cleaned not in snippets:
                    snippets.append(cleaned)
        
        return snippets[:5]  # Limit to top 5 snippets
    
    def _extract_links(self, raw_data: str) -> List[Dict[str, str]]:
        """Extract URLs and associated metadata"""
        links = []
        
        # Pattern for URLs with titles
        url_pattern = r'https://[^\s"\\]+'
        urls = re.findall(url_pattern, raw_data)
        
        # Clean and deduplicate URLs
        seen_urls = set()
        for url in urls:
            # Clean up URL
            clean_url = url.replace('\\u003d', '=').replace('\\u0026', '&')
            
            if clean_url not in seen_urls and len(clean_url) > 10:
                seen_urls.add(clean_url)
                
                # Try to find associated title/description
                title = self._find_url_title(raw_data, url)
                source = self._extract_source_from_url(clean_url)
                
                links.append({
                    'url': clean_url,
                    'title': title,
                    'source': source
                })
        
        return links[:15]  # Limit to reasonable number
    
    def _find_url_title(self, raw_data: str, url: str) -> str:
        """Find title/description associated with a URL"""
        # Look for text near the URL
        url_pos = raw_data.find(url)
        if url_pos == -1:
            return "No title found"
        
        # Search around the URL for quoted text that might be a title
        search_start = max(0, url_pos - 200)
        search_end = min(len(raw_data), url_pos + 200)
        context = raw_data[search_start:search_end]
        
        # Look for quoted strings that might be titles
        title_patterns = [
            r'"([^"]{20,100})"',  # Medium length quoted text
            r'"([^"]*(?:news|report|article|video)[^"]*)"'  # News-related text
        ]
        
        for pattern in title_patterns:
            matches = re.findall(pattern, context, re.IGNORECASE)
            for match in matches:
                if len(match) > 10 and match not in url:
                    return match[:100] + "..." if len(match) > 100 else match
        
        return "Title not found"
    
    def _extract_source_from_url(self, url: str) -> str:
        """Extract source name from URL"""
        # Common patterns for extracting source
        domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
        if domain_match:
            domain = domain_match.group(1)
            
            # Clean up common domain patterns
            if 'youtube.com' in domain:
                return 'YouTube'
            elif 'news.google.com' in domain:
                return 'Google News'
            elif 'apnews.com' in domain:
                return 'AP News'
            elif 'reuters.com' in domain:
                return 'Reuters'
            elif 'bbc.' in domain:
                return 'BBC'
            elif 'cnn.com' in domain:
                return 'CNN'
            else:
                return domain.replace('www.', '').split('.')[0].title()
        
        return "Unknown Source"

# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    st.markdown('<h1 class="main-header">🔍 Gemini Data Extractor</h1>', unsafe_allow_html=True)
    st.markdown("**Extract queries, entities, text snippets, and links from Gemini network responses**")
    
    # Show NLP status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"🧠 **BERT**: {'✅' if TRANSFORMERS_AVAILABLE else '❌'}")
    with col2:
        st.info(f"🔤 **spaCy**: {'✅' if SPACY_AVAILABLE else '❌'}")
    with col3:
        st.info("🔧 **Patterns**: ✅ Always Available")
    
    # Initialize extractor
    if 'extractor' not in st.session_state:
        st.session_state.extractor = GeminiDataExtractor()
    
    # Sidebar for NLP options
    with st.sidebar:
        st.header("🧠 Entity Extraction Options")
        
        use_nlp = st.checkbox(
            "Use Advanced NLP Models", 
            value=True,
            help="Use BERT/spaCy for better entity extraction (vs basic patterns)"
        )
        
        if use_nlp:
            st.success("🎯 **High Accuracy Mode**\nUsing BERT/spaCy for entity extraction")
        else:
            st.info("⚡ **Fast Mode**\nUsing basic pattern matching")
        
        st.markdown("---")
        st.subheader("📊 What Gets Extracted")
        st.markdown("""
        **🔍 Queries:**
        - Original search query
        - Suggested queries
        
        **💬 Text Snippets:**
        - Main response content
        - News summaries
        
        **🏷️ Entities (NLP Enhanced):**
        - People, Organizations
        - Locations, Dates, Money
        - With confidence scores
        
        **🔗 Links:**
        - URLs with titles
        - Source identification
        """)
    
    # Input section
    st.header("📥 Input Gemini Data")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        raw_input = st.text_area(
            "Paste your raw Gemini network response data:",
            height=200,
            placeholder='[["wrb.fr","hNvQHb","[[[[...your raw gemini response...]]]]"]]',
            help="Copy the raw network response from your browser's developer tools"
        )
    
    with col2:
        st.info("💡 **How to get raw data:**\n1. Open browser dev tools (F12)\n2. Go to Network tab\n3. Use Gemini\n4. Find the response\n5. Copy raw data")
        
        # Sample data button
        if st.button("📰 Load Sample Data", use_container_width=True):
            sample_data = '''[["wrb.fr","hNvQHb","[[[[\"c_6b866f1473e8f36e\",\"r_78b9bb8005ed91b1\"],null,[[\"latest news\"],3,null,0,\"71c2d248d3b102ff\",0],[[[\"rc_e706ef2eaa99080f\",[\"The latest news on June 24, 2025, is dominated by escalating tensions in the Middle East, primarily between the United States, Israel, and Iran. President Trump has announced that military strikes have totally obliterated three Iranian nuclear facilities at Fordow, Natanz, and Isfahan. Iran has launched retaliatory missile attacks on a US military base in Qatar. The international community is urging de-escalation, with the United Nations Security Council calling an emergency session. Russia and China have warned that continued US attacks on Iran risk triggering a broader global conflict. In other news, the FIFA Club World Cup 2025 continues with FC Porto facing Al Ahly at MetLife Stadium in New Jersey. A nationwide boycott of McDonald's, organized by The People's Union USA, has begun protesting alleged low wages of $12 per hour and tax avoidance of $2.3 billion.\",\"https://www.youtube.com/watch?v=ogCEiScuBJM\",\"Donald Trump said US strikes totally obliterated Iran's nuclear enrichment facilities. #BBCNews - YouTube\",\"https://apnews.com/article/israel-iran-war-nuclear-trump-bomber-news\",\"Iran launches missiles at US military base in Qatar in retaliation\"]]]],[[\"latest news headlines today\",1],[\"breaking news June 24 2025\",1],[\"What are the top 10 news headlines of today?\",4],[\"Iran nuclear facilities latest updates\",2]],null,\"rc_e706ef2eaa99080f\"]]]]'''
            st.session_state.sample_data = sample_data
            st.success("Sample data loaded!")
    
    # Use sample data if loaded
    if hasattr(st.session_state, 'sample_data') and not raw_input.strip():
        raw_input = st.session_state.sample_data
    
    # Process button
    if raw_input.strip():
        # Store raw data for debugging
        st.session_state.last_raw_data = raw_input
        
        if st.button("🔍 Extract Data", type="primary", use_container_width=True):
            with st.spinner("Extracting data with NLP models..."):
                try:
                    extracted_data = st.session_state.extractor.extract_from_gemini_raw(raw_input, use_nlp=use_nlp)
                    st.session_state.extracted_data = extracted_data
                    
                    # Show extraction summary
                    nlp_entities = len([e for e in extracted_data.entities if e['source'] in ['BERT', 'spaCy']])
                    pattern_entities = len([e for e in extracted_data.entities if e['source'] in ['patterns', 'basic_patterns']])
                    
                    st.success(f"✅ Data extracted successfully!")
                    st.info(f"🎯 **Entities Found**: {len(extracted_data.entities)} total ({nlp_entities} from NLP models, {pattern_entities} from patterns)")
                    
                    # Show quick preview
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📝 Text Snippets", len(extracted_data.text_snippets))
                    with col2:
                        st.metric("🔗 Links", len(extracted_data.links))
                    with col3:
                        st.metric("💡 Suggested Queries", len(extracted_data.suggested_queries))
                    
                except Exception as e:
                    st.error(f"❌ Extraction failed: {str(e)}")
                    st.info("💡 **Troubleshooting tips:**\n- Check if raw data is complete\n- Try the sample data first\n- Make sure data includes the full Gemini response")
    
    # Results section
    if hasattr(st.session_state, 'extracted_data'):
        data = st.session_state.extracted_data
        
        st.markdown("---")
        st.header("📊 Extracted Data")
        
        # Create tabs for different data types
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Query", "💬 Text Snippets", "🏷️ Entities", "🔗 Links", "💾 Export"])
        
        with tab1:
            st.subheader("🔍 Original Query")
            
            # Better display of original query
            if data.original_query and data.original_query != "Query not found":
                st.success(f"**Found Query:** {data.original_query}")
                
                # Show query details
                st.write("**Query Details:**")
                st.write(f"• **Text**: `{data.original_query}`")
                st.write(f"• **Length**: {len(data.original_query)} characters")
                st.write(f"• **Word count**: {len(data.original_query.split())} words")
            else:
                st.error("❌ Original query not found in the raw data")
                st.info("💡 **Possible reasons:**\n- Raw data might be incomplete\n- Different Gemini response format\n- Try using the sample data to test")
                
                # Debug info
                with st.expander("🔧 Debug Information"):
                    st.write("**Raw data preview (first 200 characters):**")
                    if hasattr(st.session_state, 'last_raw_data'):
                        preview = str(st.session_state.get('last_raw_data', ''))[:200]
                        st.code(preview)
                    else:
                        st.write("No raw data available")
            
            st.markdown("---")
            
            st.subheader("💡 All Suggested Queries")
            if data.suggested_queries:
                st.info(f"Found {len(data.suggested_queries)} suggested queries from Gemini:")
                
                # Display queries in a more readable format
                for i, query in enumerate(data.suggested_queries, 1):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"**{i}.** {query}")
                    with col2:
                        query_type = "❓" if any(word in query.lower() for word in ['what', 'how', 'when', 'where', 'why', 'who', '?']) else "📰" if any(word in query.lower() for word in ['news', 'latest', 'breaking']) else "🔍"
                        st.write(query_type)
                
                # Show categorized queries
                question_queries = [q for q in data.suggested_queries if any(word in q.lower() for word in ['what', 'how', 'when', 'where', 'why', 'who', '?'])]
                news_queries = [q for q in data.suggested_queries if any(word in q.lower() for word in ['news', 'latest', 'breaking', 'today', 'headlines'])]
                
                if question_queries or news_queries:
                    st.markdown("**📊 Query Categories:**")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if question_queries:
                            st.write(f"❓ **Questions**: {len(question_queries)}")
                            for q in question_queries[:3]:
                                st.caption(f"• {q}")
                            if len(question_queries) > 3:
                                st.caption(f"... and {len(question_queries) - 3} more")
                    
                    with col2:
                        if news_queries:
                            st.write(f"📰 **News-related**: {len(news_queries)}")
                            for q in news_queries[:3]:
                                st.caption(f"• {q}")
                            if len(news_queries) > 3:
                                st.caption(f"... and {len(news_queries) - 3} more")
                        
            else:
                st.warning("No suggested queries found")
                st.info("💡 **Troubleshooting:**\n- Raw data might not contain query suggestions\n- Different Gemini response format\n- Try the sample data to see expected format")
        
        with tab2:
            st.subheader("💬 Text Snippets")
            if data.text_snippets:
                st.info(f"Found {len(data.text_snippets)} text snippets from Gemini response:")
                
                for i, snippet in enumerate(data.text_snippets, 1):
                    with st.expander(f"Snippet {i} ({len(snippet)} characters)", expanded=i==1):
                        st.markdown(f'<div class="snippet-preview">{snippet}</div>', unsafe_allow_html=True)
                        
                        # Show snippet stats
                        word_count = len(snippet.split())
                        st.caption(f"📊 {len(snippet)} characters, {word_count} words")
            else:
                st.warning("No text snippets found")
                st.info("💡 The raw data might not contain substantial text content")
        
        with tab3:
            st.subheader("🏷️ Extracted Entities with NLP")
            if data.entities:
                # Show model breakdown
                model_breakdown = {}
                for entity in data.entities:
                    model = entity['source']
                    if model not in model_breakdown:
                        model_breakdown[model] = []
                    model_breakdown[model].append(entity)
                
                # Display model stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    bert_count = len(model_breakdown.get('BERT', []))
                    st.metric("🧠 BERT Entities", bert_count)
                with col2:
                    spacy_count = len(model_breakdown.get('spaCy', []))
                    st.metric("🔤 spaCy Entities", spacy_count)
                with col3:
                    pattern_count = len(model_breakdown.get('patterns', [])) + len(model_breakdown.get('basic_patterns', []))
                    st.metric("🔧 Pattern Entities", pattern_count)
                with col4:
                    avg_confidence = sum(e['confidence'] for e in data.entities) / len(data.entities)
                    st.metric("🎯 Avg Confidence", f"{avg_confidence:.3f}")
                
                st.markdown("---")
                
                # Group entities by type
                entities_by_type = {}
                for entity in data.entities:
                    entity_type = entity['type']
                    if entity_type not in entities_by_type:
                        entities_by_type[entity_type] = []
                    entities_by_type[entity_type].append(entity)
                
                # Display entities by type
                for entity_type, entity_list in entities_by_type.items():
                    with st.expander(f"📂 {entity_type} ({len(entity_list)} entities)", expanded=True):
                        # Sort by confidence
                        entity_list.sort(key=lambda x: x['confidence'], reverse=True)
                        
                        for entity in entity_list:
                            # Different styling based on source
                            css_class = "nlp-entity" if entity['source'] in ['BERT', 'spaCy'] else "entity-item"
                            
                            confidence_color = "🟢" if entity['confidence'] > 0.9 else "🟡" if entity['confidence'] > 0.7 else "🟠"
                            
                            # Use columns instead of HTML to avoid styling issues
                            col1, col2, col3 = st.columns([3, 1, 1])
                            
                            with col1:
                                st.write(f"**{entity['text']}**")
                                if entity.get('context'):
                                    st.caption(f"💬 {entity['context'][:60]}...")
                            
                            with col2:
                                st.write(f"{confidence_color} {entity['confidence']:.3f}")
                            
                            with col3:
                                st.write(f"🤖 {entity['source']}")
                            
                            st.markdown("---")
            else:
                st.info("No entities found")
        
        with tab4:
            st.subheader("🔗 Extracted Links")
            if data.links:
                for i, link in enumerate(data.links, 1):
                    with st.expander(f"Link {i}: {link['source']}", expanded=False):
                        st.write(f"**Title:** {link['title']}")
                        st.write(f"**Source:** {link['source']}")
                        st.write(f"**URL:** {link['url']}")
                        st.markdown(f"[🔗 Open Link]({link['url']})")
            else:
                st.info("No links found")
        
        with tab5:
            st.subheader("💾 Export Data")
            
            # Create export data with JSON-serializable objects
            export_data = {
                'extraction_timestamp': datetime.now().isoformat(),
                'extraction_method': 'NLP-enhanced' if use_nlp else 'pattern-only',
                'models_used': list(set(e['source'] for e in data.entities)),
                'original_query': data.original_query,
                'suggested_queries': data.suggested_queries,
                'text_snippets': data.text_snippets,
                'entities': [
                    {
                        'text': e['text'],
                        'type': e['type'],
                        'confidence': float(e['confidence']),  # Ensure float
                        'source': e['source'],
                        'start': int(e.get('start', 0)),  # Ensure int
                        'end': int(e.get('end', 0)),  # Ensure int
                        'context': e.get('context', '')
                    } for e in data.entities
                ],
                'links': [
                    {
                        'url': link['url'],
                        'title': link['title'],
                        'source': link['source']
                    } for link in data.links
                ],
                'summary': {
                    'total_snippets': len(data.text_snippets),
                    'total_entities': len(data.entities),
                    'total_links': len(data.links),
                    'entity_types': len(set(e['type'] for e in data.entities)),
                    'nlp_entities': len([e for e in data.entities if e['source'] in ['BERT', 'spaCy']]),
                    'pattern_entities': len([e for e in data.entities if e['source'] in ['patterns', 'basic_patterns']]),
                    'avg_confidence': float(sum(e['confidence'] for e in data.entities) / len(data.entities)) if data.entities else 0.0
                }
            }
            
            # JSON export
            json_data = json.dumps(export_data, indent=2)
            st.download_button(
                "📄 Download Complete JSON",
                json_data,
                f"gemini_extraction_nlp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json",
                use_container_width=True,
                help="Complete extraction data with NLP metadata"
            )
            
            # Enhanced CSV export for entities with NLP data
            if data.entities:
                entities_df = pd.DataFrame(data.entities)
                csv_data = entities_df.to_csv(index=False)
                st.download_button(
                    "📊 Download Entities CSV (with NLP data)",
                    csv_data,
                    f"gemini_entities_nlp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True,
                    help="Entities with confidence scores, sources, and context"
                )
            
            # CSV export for links
            if data.links:
                links_df = pd.DataFrame(data.links)
                csv_links = links_df.to_csv(index=False)
                st.download_button(
                    "🔗 Download Links CSV",
                    csv_links,
                    f"gemini_links_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            # Enhanced summary report with NLP insights
            model_stats = {}
            for entity in data.entities:
                model = entity['source']
                if model not in model_stats:
                    model_stats[model] = {'count': 0, 'avg_conf': 0, 'entities': []}
                model_stats[model]['count'] += 1
                model_stats[model]['entities'].append(entity)
            
            for model in model_stats:
                confidences = [e['confidence'] for e in model_stats[model]['entities']]
                model_stats[model]['avg_conf'] = sum(confidences) / len(confidences)
            
            summary_report = f"""# Gemini Data Extraction Report (NLP-Enhanced)
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Extraction Method: {'NLP-Enhanced' if use_nlp else 'Pattern-Only'}

## Original Query
{data.original_query}

## Summary Statistics
- Text Snippets: {len(data.text_snippets)}
- Total Entities: {len(data.entities)}
- Links Found: {len(data.links)}
- Suggested Queries: {len(data.suggested_queries)}
- Average Confidence: {sum(e['confidence'] for e in data.entities) / len(data.entities) if data.entities else 0:.3f}

## Model Performance
{chr(10).join([f"- {model}: {stats['count']} entities (avg confidence: {stats['avg_conf']:.3f})" for model, stats in model_stats.items()])}

## Entity Type Breakdown
{chr(10).join([f"- {entity_type}: {len([e for e in data.entities if e['type'] == entity_type])}" for entity_type in sorted(set(e['type'] for e in data.entities))])}

## High-Confidence Entities (>0.9)
{chr(10).join([f"- {e['text']} ({e['type']}) - {e['confidence']:.3f} via {e['source']}" for e in sorted(data.entities, key=lambda x: x['confidence'], reverse=True) if e['confidence'] > 0.9][:10])}

## Suggested Queries
{chr(10).join([f"- {q}" for q in data.suggested_queries])}

## Top Sources
{chr(10).join([f"- {link['source']}: {link['title'][:50]}..." for link in data.links[:5]])}
"""
            
            st.download_button(
                "📋 Download Enhanced Report",
                summary_report,
                f"gemini_nlp_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "text/plain",
                use_container_width=True,
                help="Detailed report with NLP analysis insights"
            )
            
            # Show extraction summary with model comparison
            st.subheader("📈 NLP Extraction Summary")
            
            if data.entities:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Model Performance:**")
                    for model, stats in model_stats.items():
                        conf_emoji = "🟢" if stats['avg_conf'] > 0.9 else "🟡" if stats['avg_conf'] > 0.7 else "🟠"
                        st.write(f"• **{model}**: {stats['count']} entities {conf_emoji} ({stats['avg_conf']:.3f} avg)")
                
                with col2:
                    st.write("**Entity Quality:**")
                    high_conf = len([e for e in data.entities if e['confidence'] > 0.9])
                    med_conf = len([e for e in data.entities if 0.7 < e['confidence'] <= 0.9])
                    low_conf = len([e for e in data.entities if e['confidence'] <= 0.7])
                    
                    st.write(f"• **High confidence (>90%)**: {high_conf} entities 🟢")
                    st.write(f"• **Medium confidence (70-90%)**: {med_conf} entities 🟡")
                    st.write(f"• **Lower confidence (≤70%)**: {low_conf} entities 🟠")
                
                # Show top entities by confidence
                st.write("**🏆 Top Entities by Confidence:**")
                top_entities = sorted(data.entities, key=lambda x: x['confidence'], reverse=True)[:5]
                for i, entity in enumerate(top_entities, 1):
                    st.write(f"{i}. **{entity['text']}** ({entity['type']}) - {entity['confidence']:.3f} via {entity['source']}")

if __name__ == "__main__":
    main()
