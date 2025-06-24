import streamlit as st
import json
import re
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any

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
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .extraction-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
    .entity-item {
        background: #e3f2fd;
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-radius: 0.3rem;
        border-left: 3px solid #1976d2;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ExtractedData:
    original_query: str
    suggested_queries: List[str]
    text_snippets: List[str]
    links: List[Dict[str, str]]
    entities: List[Dict[str, str]]

# ============================================================================
# GEMINI DATA EXTRACTOR
# ============================================================================

class GeminiDataExtractor:
    """Simple extractor focused on queries, entities, text snippets, and links"""
    
    def __init__(self):
        # Simple entity patterns for basic extraction
        self.entity_patterns = {
            'PERSON': r'\b(?:President|Prime Minister|CEO|Director|Mr\.|Mrs\.|Dr\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|Donald Trump|Joe Biden|Vladimir Putin|Xi Jinping',
            'ORGANIZATION': r'\b(?:UN|NATO|FIFA|WHO|Google|Apple|Microsoft|Amazon|McDonald\'s|Tesla)\b',
            'LOCATION': r'\b(?:United States|US|USA|Iran|Israel|China|Russia|Middle East|Europe|Asia)\b',
            'DATE': r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}|\b\d{1,2}/\d{1,2}/\d{4}\b',
            'MONEY': r'\$\d+(?:,\d{3})*(?:\.\d{2})?(?:\s*(?:million|billion|trillion))?'
        }
    
    def extract_from_gemini_raw(self, raw_data: str) -> ExtractedData:
        """Extract all requested data from raw Gemini response"""
        
        # Extract original query
        original_query = self._extract_original_query(raw_data)
        
        # Extract suggested queries
        suggested_queries = self._extract_suggested_queries(raw_data)
        
        # Extract text snippets
        text_snippets = self._extract_text_snippets(raw_data)
        
        # Extract links
        links = self._extract_links(raw_data)
        
        # Extract entities from text snippets
        entities = self._extract_entities_from_text(text_snippets)
        
        return ExtractedData(
            original_query=original_query,
            suggested_queries=suggested_queries,
            text_snippets=text_snippets,
            links=links,
            entities=entities
        )
    
    def _extract_original_query(self, raw_data: str) -> str:
        """Extract the original search query"""
        patterns = [
            r'\["([^"]+)"\],3,null,0',  # Main query pattern
            r'\[\["([^"]+)"\]',         # Alternative pattern
            r'"([^"]*latest news[^"]*)"'  # News-specific pattern
        ]
        
        for pattern in patterns:
            match = re.search(pattern, raw_data, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return "Query not found"
    
    def _extract_suggested_queries(self, raw_data: str) -> List[str]:
        """Extract suggested/related queries"""
        suggested = []
        
        # Pattern for suggested queries section
        pattern = r'\[\["([^"]+)",\d+\]'
        matches = re.findall(pattern, raw_data)
        
        # Common suggested queries in Gemini responses
        for match in matches:
            if any(word in match.lower() for word in ['news', 'latest', 'today', 'what', 'how', 'when', 'where']):
                if len(match) > 5 and match not in suggested:  # Avoid duplicates and too short queries
                    suggested.append(match)
        
        # Limit to reasonable number
        return suggested[:10]
    
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
    
    def _extract_entities_from_text(self, text_snippets: List[str]) -> List[Dict[str, str]]:
        """Extract entities from the text snippets"""
        entities = []
        all_text = " ".join(text_snippets)
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            for match in matches:
                if match and len(match.strip()) > 2:
                    entity_text = match.strip()
                    # Avoid duplicates
                    if not any(e['text'].lower() == entity_text.lower() for e in entities):
                        entities.append({
                            'text': entity_text,
                            'type': entity_type,
                            'source': 'pattern_matching'
                        })
        
        return entities

# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    st.markdown('<h1 class="main-header">🔍 Gemini Data Extractor</h1>', unsafe_allow_html=True)
    st.markdown("**Extract queries, entities, text snippets, and links from Gemini network responses**")
    
    # Initialize extractor
    if 'extractor' not in st.session_state:
        st.session_state.extractor = GeminiDataExtractor()
    
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
            sample_data = '''[["wrb.fr","hNvQHb","[[[[\"c_6b866f1473e8f36e\",\"r_78b9bb8005ed91b1\"],null,[[\"latest news\"],3,null,0,\"71c2d248d3b102ff\",0],[[[\"rc_e706ef2eaa99080f\",[\"The latest news on June 24, 2025, is dominated by escalating tensions in the Middle East, primarily between the US, Israel, and Iran. President Trump has stated these strikes totally obliterated the facilities.\",\"https://www.youtube.com/watch?v=ogCEiScuBJM\",\"Donald Trump said US strikes totally obliterated Iran's nuclear enrichment facilities. #BBCNews - YouTube\"]]]],[[\"latest news headlines today\",1],[\"breaking news June 24 2025\",1],[\"What are the top 10 news headlines of today?\",4]],null,\"rc_e706ef2eaa99080f\"]]]]'''
            st.session_state.sample_data = sample_data
            st.success("Sample data loaded!")
    
    # Use sample data if loaded
    if hasattr(st.session_state, 'sample_data') and not raw_input.strip():
        raw_input = st.session_state.sample_data
    
    # Process button
    if raw_input.strip():
        if st.button("🔍 Extract Data", type="primary", use_container_width=True):
            with st.spinner("Extracting data..."):
                try:
                    extracted_data = st.session_state.extractor.extract_from_gemini_raw(raw_input)
                    st.session_state.extracted_data = extracted_data
                    st.success("✅ Data extracted successfully!")
                except Exception as e:
                    st.error(f"❌ Extraction failed: {str(e)}")
    
    # Results section
    if hasattr(st.session_state, 'extracted_data'):
        data = st.session_state.extracted_data
        
        st.markdown("---")
        st.header("📊 Extracted Data")
        
        # Create tabs for different data types
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Query", "💬 Text Snippets", "🏷️ Entities", "🔗 Links", "💾 Export"])
        
        with tab1:
            st.subheader("🔍 Original Query")
            st.markdown(f'<div class="extraction-section"><h4>"{data.original_query}"</h4></div>', unsafe_allow_html=True)
            
            st.subheader("💡 Suggested Queries")
            if data.suggested_queries:
                for i, query in enumerate(data.suggested_queries, 1):
                    st.write(f"{i}. {query}")
            else:
                st.info("No suggested queries found")
        
        with tab2:
            st.subheader("💬 Text Snippets")
            if data.text_snippets:
                for i, snippet in enumerate(data.text_snippets, 1):
                    with st.expander(f"Snippet {i} ({len(snippet)} characters)", expanded=i==1):
                        st.write(snippet)
            else:
                st.info("No text snippets found")
        
        with tab3:
            st.subheader("🏷️ Extracted Entities")
            if data.entities:
                # Group entities by type
                entities_by_type = {}
                for entity in data.entities:
                    entity_type = entity['type']
                    if entity_type not in entities_by_type:
                        entities_by_type[entity_type] = []
                    entities_by_type[entity_type].append(entity['text'])
                
                for entity_type, entity_list in entities_by_type.items():
                    with st.expander(f"📂 {entity_type} ({len(entity_list)} entities)", expanded=True):
                        for entity_text in entity_list:
                            st.markdown(f'<div class="entity-item">• {entity_text}</div>', unsafe_allow_html=True)
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
            
            # Create export data
            export_data = {
                'extraction_timestamp': datetime.now().isoformat(),
                'original_query': data.original_query,
                'suggested_queries': data.suggested_queries,
                'text_snippets': data.text_snippets,
                'entities': data.entities,
                'links': data.links,
                'summary': {
                    'total_snippets': len(data.text_snippets),
                    'total_entities': len(data.entities),
                    'total_links': len(data.links),
                    'entity_types': len(set(e['type'] for e in data.entities))
                }
            }
            
            # JSON export
            json_data = json.dumps(export_data, indent=2)
            st.download_button(
                "📄 Download JSON",
                json_data,
                f"gemini_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json",
                use_container_width=True
            )
            
            # CSV export for entities
            if data.entities:
                entities_df = pd.DataFrame(data.entities)
                csv_data = entities_df.to_csv(index=False)
                st.download_button(
                    "📊 Download Entities CSV",
                    csv_data,
                    f"gemini_entities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
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
            
            # Summary report
            summary_report = f"""# Gemini Data Extraction Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Original Query
{data.original_query}

## Summary
- Text Snippets: {len(data.text_snippets)}
- Entities Found: {len(data.entities)}
- Links Found: {len(data.links)}
- Suggested Queries: {len(data.suggested_queries)}

## Suggested Queries
{chr(10).join([f"- {q}" for q in data.suggested_queries])}

## Entity Breakdown
{chr(10).join([f"- {entity_type}: {len([e for e in data.entities if e['type'] == entity_type])}" for entity_type in set(e['type'] for e in data.entities)])}

## Top Links
{chr(10).join([f"- {link['source']}: {link['title'][:50]}..." for link in data.links[:5]])}
"""
            
            st.download_button(
                "📋 Download Summary Report",
                summary_report,
                f"gemini_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "text/plain",
                use_container_width=True
            )
            
            # Show summary
            st.subheader("📈 Extraction Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📝 Text Snippets", len(data.text_snippets))
            with col2:
                st.metric("🏷️ Entities", len(data.entities))
            with col3:
                st.metric("🔗 Links", len(data.links))
            with col4:
                st.metric("💡 Suggested Queries", len(data.suggested_queries))

if __name__ == "__main__":
    main()
