import streamlit as st
import json
import re
import pandas as pd
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import warnings
import io
import base64
from datetime import datetime
warnings.filterwarnings('ignore')

# Import optional packages with fallbacks
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

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
    page_title="🧠 Advanced Gemini Parser",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-container {
        background: linear-gradient(90deg, #E3F2FD 0%, #BBDEFB 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1E88E5;
    }
    .entity-container {
        background: #F8F9FA;
        padding: 0.8rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
        border-left: 3px solid #28A745;
    }
    .stAlert > div {
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Entity:
    text: str
    label: str
    start_pos: int
    end_pos: int
    confidence: float
    context: str = ""
    source_model: str = ""
    
    def to_dict(self):
        return asdict(self)

# ============================================================================
# BASIC PATTERN EXTRACTOR (ALWAYS AVAILABLE)
# ============================================================================

class BasicEntityExtractor:
    """Pattern-based entity extraction that works without external libraries"""
    
    def __init__(self):
        self.patterns = {
            'PERSON': [
                r'\b(?:President|Prime Minister|Chancellor|Vice President|Secretary|Minister)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
                r'\b(?:Donald Trump|Joe Biden|Vladimir Putin|Xi Jinping|Emmanuel Macron|Angela Merkel)\b',
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?=\s+(?:said|announced|stated|declared|reported))'
            ],
            'COUNTRY': [
                r'\b(?:United States|US|USA|Iran|Israel|Qatar|Russia|China|Philippines|Nigeria|UK|Britain|France|Germany|Japan|India|Brazil|Canada|Mexico|Australia)\b'
            ],
            'ORGANIZATION': [
                r'\b(?:UN|NATO|FIFA|WHO|IMF|World Bank|European Union|EU|OPEC)\b',
                r'\b(?:Google|Apple|Microsoft|Amazon|Meta|Tesla|McDonald\'s|Coca-Cola|Nike)\b',
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|Company|Corporation|Ltd|Organization|Association|Foundation)\b'
            ],
            'LOCATION': [
                r'\b(?:Middle East|Asia|Europe|Africa|Americas|Pacific|Atlantic|Mediterranean)\b',
                r'\b(?:New York|Washington|London|Paris|Berlin|Tokyo|Beijing|Moscow|Tehran|Tel Aviv)\b',
                r'\b[A-Z][a-z]+\s+(?:Stadium|Airport|Base|Center|Building|Hospital|University|School)\b'
            ],
            'DATE': [
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
                r'\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b',
                r'\b(?:today|yesterday|tomorrow|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b'
            ],
            'EVENT': [
                r'\b(?:World Cup|Olympics|Paralympic|NATO Summit|G7|G20|World Economic Forum|COP\d+)\b',
                r'\b[A-Z][a-z]+\s+(?:Conference|Summit|Meeting|Championship|Cup|Games|Festival|Concert)\s*\d{4}?\b'
            ],
            'MONEY': [
                r'\$\d+(?:,\d{3})*(?:\.\d{2})?(?:\s*(?:million|billion|trillion|thousand))?',
                r'\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars|euros|pounds|yen|yuan)\b'
            ],
            'FACILITY': [
                r'\b(?:Fordow|Natanz|Isfahan|Bushehr|Arak)\b',
                r'\b[A-Z][a-z]+\s+(?:nuclear|power|military|air|naval)\s+(?:facility|plant|base|station)\b'
            ]
        }
    
    def extract_entities(self, text):
        """Extract entities using regex patterns"""
        entities = []
        
        for label, patterns in self.patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    start_ctx = max(0, match.start() - 40)
                    end_ctx = min(len(text), match.end() + 40)
                    context = text[start_ctx:end_ctx].replace('\n', ' ').strip()
                    
                    entity = Entity(
                        text=match.group().strip(),
                        label=label,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.75,
                        context=context,
                        source_model="pattern_matching"
                    )
                    entities.append(entity)
        
        # Remove duplicates and overlaps
        return self._remove_duplicates(entities)
    
    def _remove_duplicates(self, entities):
        """Remove duplicate and overlapping entities"""
        if not entities:
            return []
        
        # Sort by position
        entities.sort(key=lambda x: x.start_pos)
        
        unique = []
        for entity in entities:
            # Check for exact duplicates
            if any(e.text.lower() == entity.text.lower() and e.label == entity.label 
                   for e in unique):
                continue
            
            # Check for overlaps
            overlap = False
            for i, existing in enumerate(unique):
                if (entity.start_pos < existing.end_pos and 
                    entity.end_pos > existing.start_pos):
                    # Keep the longer entity or higher confidence
                    if len(entity.text) > len(existing.text) or entity.confidence > existing.confidence:
                        unique[i] = entity
                    overlap = True
                    break
            
            if not overlap:
                unique.append(entity)
        
        return unique

# ============================================================================
# ADVANCED PARSER (WITH FALLBACKS)
# ============================================================================

@st.cache_resource
class StreamlitGeminiParser:
    """Gemini parser with multiple fallback options"""
    
    def __init__(self):
        self.ner_pipeline = None
        self.nlp = None
        self.basic_extractor = BasicEntityExtractor()
        self.is_initialized = False
        self.model_type = None
    
    def initialize(self, model_choice="basic-only"):
        """Initialize with fallback to basic extraction"""
        
        if self.is_initialized and self.model_type == model_choice:
            return True
        
        try:
            # Try transformer models first
            if model_choice in ["bert-base", "bert-large", "distilbert"] and TRANSFORMERS_AVAILABLE:
                models = {
                    "bert-base": "dslim/bert-base-NER",
                    "bert-large": "dbmdz/bert-large-cased-finetuned-conll03-english",
                    "distilbert": "dslim/distilbert-NER"
                }
                
                try:
                    self.ner_pipeline = pipeline(
                        "ner",
                        model=models[model_choice],
                        aggregation_strategy="simple",
                        device=-1  # Force CPU for cloud deployment
                    )
                    self.model_type = model_choice
                    self.is_initialized = True
                    return True
                except Exception as e:
                    st.warning(f"Transformer model failed: {str(e)[:100]}...")
            
            # Try spaCy
            if model_choice == "spacy-only" and SPACY_AVAILABLE:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    self.model_type = "spacy-only"
                    self.is_initialized = True
                    return True
                except Exception as e:
                    st.warning(f"spaCy model failed: {str(e)[:100]}...")
            
            # Fallback to basic extraction
            self.model_type = "basic-only"
            self.is_initialized = True
            return True
            
        except Exception as e:
            st.error(f"Initialization error: {str(e)[:100]}...")
            self.model_type = "basic-only"
            self.is_initialized = True
            return True
    
    def extract_entities(self, text):
        """Extract entities with fallback chain"""
        
        if not self.is_initialized:
            return []
        
        entities = []
        
        try:
            # Try transformer pipeline
            if self.ner_pipeline:
                results = self.ner_pipeline(text)
                for result in results:
                    start_ctx = max(0, result["start"] - 50)
                    end_ctx = min(len(text), result["end"] + 50)
                    context = text[start_ctx:end_ctx].replace('\n', ' ')
                    
                    entity = Entity(
                        text=result["word"],
                        label=result["entity_group"],
                        start_pos=result["start"],
                        end_pos=result["end"],
                        confidence=result["score"],
                        context=context,
                        source_model="transformer"
                    )
                    entities.append(entity)
                
                return self._remove_duplicates(entities)
            
            # Try spaCy
            elif self.nlp:
                doc = self.nlp(text)
                for ent in doc.ents:
                    start_ctx = max(0, ent.start_char - 50)
                    end_ctx = min(len(text), ent.end_char + 50)
                    context = text[start_ctx:end_ctx].replace('\n', ' ')
                    
                    entity = Entity(
                        text=ent.text,
                        label=ent.label_,
                        start_pos=ent.start_char,
                        end_pos=ent.end_char,
                        confidence=0.85,
                        context=context,
                        source_model="spacy"
                    )
                    entities.append(entity)
                
                return self._remove_duplicates(entities)
            
            # Fallback to basic extraction
            else:
                return self.basic_extractor.extract_entities(text)
                
        except Exception as e:
            st.warning(f"Extraction error: {str(e)[:100]}... Using basic extraction.")
            return self.basic_extractor.extract_entities(text)
    
    def _remove_duplicates(self, entities):
        """Remove overlapping entities"""
        return self.basic_extractor._remove_duplicates(entities)
    
    def extract_text_from_gemini_raw(self, raw_data):
        """Extract text from raw Gemini data"""
        try:
            # Multiple patterns to extract text
            patterns = [
                r'"(The latest news[^"]*(?:\\"[^"]*)*)"',
                r'"(Breaking news[^"]*(?:\\"[^"]*)*)"',
                r'"([^"]{300,})"',  # Any long text
                r'"([^"]*(?:President|Iran|Israel|Trump|news)[^"]*)"'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, raw_data, re.IGNORECASE)
                if match:
                    text = match.group(1)
                    # Clean up escaped characters
                    text = text.replace('\\"', '"').replace('\\n', '\n').replace('\\t', ' ')
                    if len(text) > 100:  # Minimum length check
                        return text
            
            return None
            
        except Exception as e:
            st.error(f"Text extraction error: {e}")
            return None

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_visualizations(entities):
    """Create visualizations with Plotly fallback"""
    
    if not entities:
        st.warning("No entities to visualize!")
        return
    
    df = pd.DataFrame([e.to_dict() for e in entities])
    
    if PLOTLY_AVAILABLE:
        create_plotly_charts(df)
    else:
        create_basic_charts(df)

def create_plotly_charts(df):
    """Create interactive Plotly charts"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Entity type pie chart
        type_counts = df['label'].value_counts()
        fig_pie = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="🏷️ Entity Types",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Confidence histogram
        fig_hist = px.histogram(
            df, 
            x='confidence', 
            title="🎯 Confidence Distribution",
            nbins=15,
            color_discrete_sequence=['#1E88E5']
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Entity details table
    st.subheader("📋 Entity Details")
    display_df = df[['text', 'label', 'confidence', 'source_model']].sort_values('confidence', ascending=False)
    st.dataframe(display_df, use_container_width=True)

def create_basic_charts(df):
    """Create basic Streamlit charts"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏷️ Entity Types")
        type_counts = df['label'].value_counts()
        st.bar_chart(type_counts)
    
    with col2:
        st.subheader("🎯 Confidence Distribution")
        confidence_bins = pd.cut(df['confidence'], bins=10).value_counts().sort_index()
        st.bar_chart(confidence_bins)
    
    # Entity table
    st.subheader("📋 All Entities")
    st.dataframe(df[['text', 'label', 'confidence']].sort_values('confidence', ascending=False))

def display_entity_metrics(entities):
    """Display key metrics"""
    
    if not entities:
        return
    
    df = pd.DataFrame([e.to_dict() for e in entities])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Entities", len(entities))
    
    with col2:
        st.metric("🎯 Avg Confidence", f"{df['confidence'].mean():.3f}")
    
    with col3:
        st.metric("🏷️ Unique Types", df['label'].nunique())
    
    with col4:
        high_conf = len(df[df['confidence'] > 0.8])
        st.metric("⭐ High Confidence", high_conf)

def display_entities_by_type(entities):
    """Display entities grouped by type"""
    
    if not entities:
        return
    
    # Group by type
    by_type = defaultdict(list)
    for entity in entities:
        by_type[entity.label].append(entity)
    
    # Display each type
    for entity_type, type_entities in sorted(by_type.items()):
        with st.expander(f"📂 {entity_type} ({len(type_entities)} entities)", expanded=True):
            # Sort by confidence
            sorted_entities = sorted(type_entities, key=lambda x: x.confidence, reverse=True)
            
            for entity in sorted_entities[:10]:  # Show top 10 per type
                col1, col2, col3 = st.columns([4, 1, 1])
                
                with col1:
                    st.write(f"**{entity.text}**")
                    if entity.context:
                        st.caption(f"💬 {entity.context[:80]}...")
                
                with col2:
                    st.write(f"🎯 {entity.confidence:.3f}")
                
                with col3:
                    st.write(f"🤖 {entity.source_model}")
            
            if len(type_entities) > 10:
                st.info(f"... and {len(type_entities) - 10} more entities")

# ============================================================================
# SAMPLE DATA
# ============================================================================

def get_sample_data():
    """Sample news data for testing"""
    return """The latest news on June 24, 2025, is dominated by escalating tensions in the Middle East, primarily between the United States, Israel, and Iran. President Trump has announced that military strikes have "totally obliterated" three Iranian nuclear facilities at Fordow, Natanz, and Isfahan. Iran has launched retaliatory missile attacks on a US military base in Qatar, with reports of explosions and damage. Qatar's air defense systems reportedly intercepted some of the incoming missiles. The international community is urging de-escalation, with the United Nations nuclear watchdog expressing "grave alarm" over the situation. Russia and China have warned that continued US attacks on Iran risk triggering a broader global conflict. In other news, the FIFA Club World Cup 2025 continues with FC Porto facing Al Ahly at MetLife Stadium in New Jersey. A nationwide boycott of McDonald's, organized by The People's Union USA, has begun on June 24th, protesting alleged low wages, tax avoidance, and lack of corporate accountability. In the Philippines, Vice President Sara Duterte is facing an impeachment complaint, which she has dismissed as meaningless. Nigeria has been crowned the first-ever African men's and women's flag football champions, while the Oklahoma City Thunder won the NBA championship."""

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Advanced Gemini Parser</h1>', unsafe_allow_html=True)
    st.markdown("**Extract entities from Gemini responses using NLP and pattern matching**")
    
    # Show system status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"🖥️ **Transformers**: {'✅' if TRANSFORMERS_AVAILABLE else '❌'}")
    with col2:
        st.info(f"🔤 **spaCy**: {'✅' if SPACY_AVAILABLE else '❌'}")
    with col3:
        st.info(f"📊 **Plotly**: {'✅' if PLOTLY_AVAILABLE else '❌'}")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        available_models = [("Basic Pattern Matching", "basic-only")]
        
        if TRANSFORMERS_AVAILABLE:
            available_models.extend([
                ("BERT Base (Recommended)", "bert-base"),
                ("BERT Large (Slow)", "bert-large"),
                ("DistilBERT (Fast)", "distilbert")
            ])
        
        if SPACY_AVAILABLE:
            available_models.append(("spaCy NER", "spacy-only"))
        
        model_choice = st.selectbox(
            "🤖 Select Model",
            options=available_models,
            format_func=lambda x: x[0]
        )[1]
        
        # Model info
        if model_choice == "basic-only":
            st.info("🔧 Uses regex patterns. Always available but lower accuracy.")
        elif "bert" in model_choice:
            st.info("🧠 Uses BERT transformer. High accuracy but slower.")
        elif model_choice == "spacy-only":
            st.info("⚡ Uses spaCy NER. Good balance of speed and accuracy.")
        
        # Initialize
        if 'parser' not in st.session_state:
            st.session_state.parser = StreamlitGeminiParser()
        
        if st.button("🚀 Initialize Model", type="primary"):
            with st.spinner("Loading..."):
                success = st.session_state.parser.initialize(model_choice)
                if success:
                    st.success("✅ Model Ready!")
                    st.session_state.model_ready = True
                else:
                    st.error("❌ Failed to initialize")
        
        # Status
        if hasattr(st.session_state, 'model_ready') and st.session_state.model_ready:
            st.success(f"🟢 {st.session_state.parser.model_type} ready")
        
        # Quick demo
        st.header("🎯 Quick Demo")
        if st.button("📰 Load Sample News"):
            st.session_state.sample_loaded = True
            st.success("Sample loaded! Go to Input tab.")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📝 Input", "🔍 Results", "📊 Analytics"])
    
    with tab1:
        st.header("📝 Input Data")
        
        # Input methods
        input_type = st.radio(
            "Choose input method:",
            ["📄 Plain Text", "🔗 Raw Gemini Data", "📁 File Upload"],
            horizontal=True
        )
        
        text_to_process = None
        
        if input_type == "📄 Plain Text":
            # Use sample if loaded
            default_text = ""
            if hasattr(st.session_state, 'sample_loaded') and st.session_state.sample_loaded:
                default_text = get_sample_data()
            
            text_input = st.text_area(
                "Enter text to analyze:",
                value=default_text,
                height=250,
                placeholder="Enter news text, articles, or any content to extract entities from..."
            )
            
            if text_input.strip():
                text_to_process = text_input
        
        elif input_type == "🔗 Raw Gemini Data":
            raw_input = st.text_area(
                "Paste raw Gemini network response:",
                height=200,
                placeholder='[["wrb.fr","hNvQHb","[[[[...]]]]"]]'
            )
            
            if raw_input.strip():
                if hasattr(st.session_state, 'model_ready') and st.session_state.model_ready:
                    extracted = st.session_state.parser.extract_text_from_gemini_raw(raw_input)
                    if extracted:
                        st.success(f"✅ Extracted {len(extracted)} characters")
                        with st.expander("📄 Extracted Text Preview"):
                            st.write(extracted[:500] + "..." if len(extracted) > 500 else extracted)
                        text_to_process = extracted
                    else:
                        st.error("❌ Could not extract text from raw data")
        
        else:  # File Upload
            uploaded_file = st.file_uploader(
                "Upload text file (.txt):",
                type=['txt'],
                help="Upload a text file containing content to analyze"
            )
            
            if uploaded_file:
                try:
                    file_content = uploaded_file.read().decode('utf-8')
                    st.success(f"✅ File uploaded: {uploaded_file.name} ({len(file_content)} chars)")
                    
                    # Auto-detect format
                    if '[["' in file_content or 'wrb.fr' in file_content:
                        st.info("🔍 Detected Gemini format")
                        if hasattr(st.session_state, 'model_ready') and st.session_state.model_ready:
                            extracted = st.session_state.parser.extract_text_from_gemini_raw(file_content)
                            if extracted:
                                text_to_process = extracted
                            else:
                                st.error("Could not extract text")
                    else:
                        st.info("📝 Processing as plain text")
                        text_to_process = file_content
                        
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        # Process button
        if text_to_process:
            st.markdown("---")
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.info(f"📄 Ready to process {len(text_to_process)} characters")
            
            with col2:
                if hasattr(st.session_state, 'model_ready') and st.session_state.model_ready:
                    if st.button("🧠 Extract Entities", type="primary", use_container_width=True):
                        with st.spinner("🔍 Extracting entities..."):
                            entities = st.session_state.parser.extract_entities(text_to_process)
                            
                            if entities:
                                st.session_state.entities = entities
                                st.session_state.processed_text = text_to_process
                                st.success(f"✅ Found {len(entities)} entities!")
                                st.balloons()
                            else:
                                st.warning("⚠️ No entities found")
                else:
                    st.error("❌ Initialize model first")
    
    with tab2:
        st.header("🔍 Extraction Results")
        
        if hasattr(st.session_state, 'entities') and st.session_state.entities:
            entities = st.session_state.entities
            
            # Metrics
            display_entity_metrics(entities)
            
            st.markdown("---")
            
            # Entities by type
            display_entities_by_type(entities)
            
            # Export section
            st.markdown("---")
            st.header("💾 Export Results")
            
            # Prepare export data
            df_export = pd.DataFrame([e.to_dict() for e in entities])
            csv_data = df_export.to_csv(index=False)
            json_data = json.dumps([e.to_dict() for e in entities], indent=2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    "📥 Download CSV",
                    csv_data,
                    f"gemini_entities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                st.download_button(
                    "📄 Download JSON",
                    json_data,
                    f"gemini_entities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json",
                    use_container_width=True
                )
            
        else:
            st.info("👈 Process some text in the Input tab to see results here!")
    
    with tab3:
        st.header("📊 Analytics & Visualizations")
        
        if hasattr(st.session_state, 'entities') and st.session_state.entities:
            create_visualizations(st.session_state.entities)
        else:
            st.info("👈 Extract entities first to see analytics!")

if __name__ == "__main__":
    main()
