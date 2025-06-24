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
import subprocess
import sys
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
# SPACY MODEL DOWNLOAD FUNCTION
# ============================================================================

def download_spacy_model():
    """Download spaCy model if not available"""
    if not SPACY_AVAILABLE:
        return False
    
    try:
        # Try to load the model first
        import spacy
        nlp = spacy.load("en_core_web_sm")
        return True
    except OSError:
        # Model not found, try to download
        try:
            st.info("📥 Downloading spaCy model... This may take a moment.")
            
            # Run the download command
            result = subprocess.run([
                sys.executable, "-m", "spacy", "download", "en_core_web_sm"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                st.success("✅ spaCy model downloaded successfully!")
                # Try to load again
                nlp = spacy.load("en_core_web_sm")
                return True
            else:
                st.warning("⚠️ spaCy model download failed. Using fallback methods.")
                return False
                
        except Exception as e:
            st.warning(f"⚠️ Could not download spaCy model: {str(e)[:100]}...")
            return False

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
# ENHANCED PATTERN EXTRACTOR
# ============================================================================

class EnhancedPatternExtractor:
    """Enhanced pattern-based entity extraction with more comprehensive patterns"""
    
    def __init__(self):
        self.patterns = {
            'PERSON': [
                r'\b(?:President|Prime Minister|Chancellor|Vice President|Secretary|Minister|Chairman|CEO|Director)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
                r'\b(?:Donald Trump|Joe Biden|Vladimir Putin|Xi Jinping|Emmanuel Macron|Angela Merkel|Narendra Modi|Justin Trudeau)\b',
                r'\b(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?=\s+(?:said|announced|stated|declared|reported|confirmed|denied|claimed))',
                r'\b(?:Sara Duterte|Ayatollah Ali Khamenei|Pope Francis|King Charles)\b'
            ],
            'COUNTRY': [
                r'\b(?:United States|US|USA|America|Iran|Israel|Qatar|Russia|China|Philippines|Nigeria|UK|Britain|France|Germany|Japan|India|Brazil|Canada|Mexico|Australia|Italy|Spain|Turkey|Egypt|Saudi Arabia|UAE|South Korea|North Korea)\b'
            ],
            'ORGANIZATION': [
                r'\b(?:UN|NATO|FIFA|WHO|IMF|World Bank|European Union|EU|OPEC|WTO|UNESCO|UNICEF)\b',
                r'\b(?:Google|Apple|Microsoft|Amazon|Meta|Tesla|McDonald\'s|Coca-Cola|Nike|Samsung|Sony|Toyota|Walmart)\b',
                r'\b(?:The People\'s Union USA|PDP-Laban|Oklahoma City Thunder|FC Porto|Al Ahly)\b',
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|Company|Corporation|Ltd|Organization|Association|Foundation|University|College|Hospital)\b'
            ],
            'LOCATION': [
                r'\b(?:Middle East|Asia|Europe|Africa|Americas|Pacific|Atlantic|Mediterranean|Arctic|Antarctic)\b',
                r'\b(?:New York|Washington|London|Paris|Berlin|Tokyo|Beijing|Moscow|Tehran|Tel Aviv|Dubai|Cairo|Mumbai|Sydney|Toronto|Mexico City)\b',
                r'\b(?:Fordow|Natanz|Isfahan|Bushehr|Arak|Al-Udeid Air Base|Pentagon|Camp David|Guantanamo Bay)\b',
                r'\b[A-Z][a-z]+\s+(?:Stadium|Airport|Base|Center|Building|Hospital|University|School|Bridge|Tower|Palace|Embassy)\b',
                r'\b(?:MetLife Stadium|Madison Square Garden|Wembley Stadium|Old Trafford)\b'
            ],
            'DATE': [
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
                r'\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b',
                r'\b(?:today|yesterday|tomorrow|this morning|this afternoon|tonight|last night|next week|last week|this month|next month)\b',
                r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
                r'\b(?:Q1|Q2|Q3|Q4)\s+\d{4}\b'
            ],
            'EVENT': [
                r'\b(?:World Cup|Olympics|Paralympic Games|NATO Summit|G7|G20|World Economic Forum|COP\d+|Super Bowl|World Series)\b',
                r'\b(?:FIFA Club World Cup 2025|NBA championship|Champions League|World Championship)\b',
                r'\b[A-Z][a-z]+\s+(?:Conference|Summit|Meeting|Championship|Cup|Games|Festival|Concert|Election|War|Crisis|Conflict)\s*\d{4}?\b'
            ],
            'MONEY': [
                r'\$\d+(?:,\d{3})*(?:\.\d{2})?(?:\s*(?:million|billion|trillion|thousand|M|B|T|K))?',
                r'\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars|euros|pounds|yen|yuan|rupees)\b',
                r'\b(?:€|£|¥|₹)\d+(?:,\d{3})*(?:\.\d{2})?\b'
            ],
            'FACILITY': [
                r'\b(?:Fordow|Natanz|Isfahan|Bushehr|Arak|Dimona|Yongbyon)\b',
                r'\b[A-Z][a-z]+\s+(?:nuclear|power|military|air|naval|space|research)\s+(?:facility|plant|base|station|center|complex)\b',
                r'\b(?:nuclear facilities|power plants|military bases|research centers)\b'
            ],
            'MILITARY': [
                r'\b(?:missile attacks|drone strikes|air strikes|naval blockade|cyber attacks|sanctions)\b',
                r'\b(?:nuclear weapons|ballistic missiles|cruise missiles|fighter jets|naval fleet|submarines)\b',
                r'\b(?:air defenses|missile defense|radar systems|early warning systems)\b'
            ],
            'PERCENTAGE': [
                r'\b\d+(?:\.\d+)?%\b',
                r'\b\d+(?:\.\d+)?\s*percent\b'
            ]
        }
    
    def extract_entities(self, text):
        """Extract entities using enhanced regex patterns"""
        entities = []
        
        for label, patterns in self.patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    start_ctx = max(0, match.start() - 50)
                    end_ctx = min(len(text), match.end() + 50)
                    context = text[start_ctx:end_ctx].replace('\n', ' ').strip()
                    
                    # Assign confidence based on pattern type and specificity
                    confidence = self._calculate_confidence(label, match.group(), pattern)
                    
                    entity = Entity(
                        text=match.group().strip(),
                        label=label,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=confidence,
                        context=context,
                        source_model="enhanced_patterns"
                    )
                    entities.append(entity)
        
        # Remove duplicates and overlaps
        return self._remove_duplicates(entities)
    
    def _calculate_confidence(self, label, text, pattern):
        """Calculate confidence score based on pattern specificity"""
        base_confidence = 0.75
        
        # Higher confidence for specific known entities
        high_confidence_entities = [
            'Donald Trump', 'Joe Biden', 'Vladimir Putin', 'Xi Jinping',
            'United States', 'Iran', 'Israel', 'China', 'Russia',
            'UN', 'NATO', 'FIFA', 'WHO', 'Google', 'Apple', 'Microsoft'
        ]
        
        if any(entity.lower() in text.lower() for entity in high_confidence_entities):
            return 0.90
        
        # Medium confidence for titles + names
        if re.match(r'\b(?:President|Prime Minister|CEO|Director)\s+', text):
            return 0.85
        
        # Lower confidence for generic patterns
        if len(text.split()) == 1:  # Single word entities
            return 0.70
        
        return base_confidence
    
    def _remove_duplicates(self, entities):
        """Remove duplicate and overlapping entities with improved logic"""
        if not entities:
            return []
        
        # Sort by position
        entities.sort(key=lambda x: x.start_pos)
        
        unique = []
        for entity in entities:
            # Check for exact duplicates (case insensitive)
            if any(e.text.lower() == entity.text.lower() and e.label == entity.label 
                   for e in unique):
                continue
            
            # Check for overlaps
            overlap = False
            for i, existing in enumerate(unique):
                if (entity.start_pos < existing.end_pos and 
                    entity.end_pos > existing.start_pos):
                    
                    # Keep the entity with higher confidence, or longer text if confidence is equal
                    if (entity.confidence > existing.confidence or 
                        (entity.confidence == existing.confidence and len(entity.text) > len(existing.text))):
                        unique[i] = entity
                    overlap = True
                    break
            
            if not overlap:
                unique.append(entity)
        
        return unique

# ============================================================================
# ADVANCED PARSER WITH BETTER ERROR HANDLING
# ============================================================================

@st.cache_resource
class StreamlitGeminiParser:
    """Enhanced Gemini parser with better fallback handling"""
    
    def __init__(self):
        self.ner_pipeline = None
        self.nlp = None
        self.pattern_extractor = EnhancedPatternExtractor()
        self.is_initialized = False
        self.model_type = None
        self.spacy_model_available = False
    
    def initialize(self, model_choice="enhanced-patterns"):
        """Initialize with comprehensive fallback system"""
        
        if self.is_initialized and self.model_type == model_choice:
            return True
        
        # Reset previous state
        self.ner_pipeline = None
        self.nlp = None
        self.spacy_model_available = False
        success = False
        
        try:
            # Handle enhanced-patterns choice directly
            if model_choice == "enhanced-patterns":
                self.model_type = "enhanced-patterns"
                self.is_initialized = True
                st.success("✅ Enhanced pattern matching ready!")
                return True
            
            # Try transformer models
            elif model_choice in ["bert-base", "bert-large", "distilbert"] and TRANSFORMERS_AVAILABLE:
                models = {
                    "bert-base": "dslim/bert-base-NER",
                    "bert-large": "dbmdz/bert-large-cased-finetuned-conll03-english",
                    "distilbert": "dslim/distilbert-NER"
                }
                
                try:
                    with st.spinner(f"Loading {model_choice} model..."):
                        self.ner_pipeline = pipeline(
                            "ner",
                            model=models[model_choice],
                            aggregation_strategy="simple",
                            device=-1  # Force CPU for cloud deployment
                        )
                    self.model_type = model_choice
                    self.is_initialized = True
                    success = True
                    st.success(f"✅ {model_choice} loaded successfully!")
                    return True
                except Exception as e:
                    st.warning(f"⚠️ {model_choice} failed: {str(e)[:100]}... Trying fallback.")
            
            # Try spaCy
            elif model_choice == "spacy-only" and SPACY_AVAILABLE:
                try:
                    with st.spinner("Loading spaCy model..."):
                        if download_spacy_model():
                            import spacy
                            self.nlp = spacy.load("en_core_web_sm")
                            self.model_type = "spacy-only"
                            self.is_initialized = True
                            self.spacy_model_available = True
                            success = True
                            st.success("✅ spaCy model loaded successfully!")
                            return True
                        else:
                            raise Exception("spaCy model download failed")
                except Exception as e:
                    st.warning(f"⚠️ spaCy loading failed: {str(e)[:100]}... Using fallback.")
            
            # If we get here, the requested model failed or isn't available
            # Fall back to enhanced pattern matching
            self.model_type = "enhanced-patterns"
            self.is_initialized = True
            
            if model_choice != "enhanced-patterns":
                st.info("ℹ️ Requested model not available. Using enhanced pattern matching as fallback.")
            
            st.success("✅ Enhanced pattern matching ready!")
            return True
            
        except Exception as e:
            # Final emergency fallback
            st.warning(f"Initialization error: {str(e)[:100]}...")
            self.model_type = "enhanced-patterns"
            self.is_initialized = True
            st.info("ℹ️ Using enhanced pattern matching as emergency fallback.")
            return True
    
    def extract_entities(self, text):
        """Extract entities with comprehensive fallback chain"""
        
        if not self.is_initialized:
            return []
        
        entities = []
        
        try:
            # Try transformer pipeline first
            if self.ner_pipeline:
                try:
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
                except Exception as e:
                    st.warning(f"Transformer extraction failed: {str(e)[:50]}... Using fallback.")
            
            # Try spaCy if available
            if self.nlp and self.spacy_model_available:
                try:
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
                except Exception as e:
                    st.warning(f"spaCy extraction failed: {str(e)[:50]}... Using pattern matching.")
            
            # Always fall back to pattern extraction
            return self.pattern_extractor.extract_entities(text)
                
        except Exception as e:
            st.warning(f"All extraction methods failed: {str(e)[:50]}... Using basic patterns.")
            return self.pattern_extractor.extract_entities(text)
    
    def _remove_duplicates(self, entities):
        """Remove overlapping entities"""
        return self.pattern_extractor._remove_duplicates(entities)
    
    def extract_text_from_gemini_raw(self, raw_data):
        """Enhanced text extraction from raw Gemini data"""
        try:
            # Multiple comprehensive patterns
            patterns = [
                r'"(The latest news[^"]*(?:\\"[^"]*)*)"',
                r'"(Breaking news[^"]*(?:\\"[^"]*)*)"',
                r'"(President [^"]*(?:\\"[^"]*)*)"',
                r'"([^"]*(?:Iran|Israel|Trump|Ukraine|Russia|China|Biden)[^"]*(?:\\"[^"]*)*)"',
                r'"([^"]{500,})"',  # Very long text
                r'"([^"]{200,})"',  # Moderately long text
                r'"([^"]*news[^"]*)"'  # Anything with "news"
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, raw_data, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    # Clean up escaped characters
                    text = match.replace('\\"', '"').replace('\\n', '\n').replace('\\t', ' ')
                    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
                    
                    # Minimum length and content checks
                    if (len(text) > 100 and 
                        any(word in text.lower() for word in ['news', 'president', 'country', 'war', 'conflict', 'announced', 'said'])):
                        return text
            
            return None
            
        except Exception as e:
            st.error(f"Text extraction error: {e}")
            return None

# ============================================================================
# VISUALIZATION FUNCTIONS (SAME AS BEFORE)
# ============================================================================

def create_visualizations(entities):
    if not entities:
        st.warning("No entities to visualize!")
        return
    
    df = pd.DataFrame([e.to_dict() for e in entities])
    
    if PLOTLY_AVAILABLE:
        create_plotly_charts(df)
    else:
        create_basic_charts(df)

def create_plotly_charts(df):
    col1, col2 = st.columns(2)
    
    with col1:
        type_counts = df['label'].value_counts()
        fig_pie = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="🏷️ Entity Types",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        fig_hist = px.histogram(
            df, 
            x='confidence', 
            title="🎯 Confidence Distribution",
            nbins=15,
            color_discrete_sequence=['#1E88E5']
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    st.subheader("📋 Entity Details")
    display_df = df[['text', 'label', 'confidence', 'source_model']].sort_values('confidence', ascending=False)
    st.dataframe(display_df, use_container_width=True)

def create_basic_charts(df):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏷️ Entity Types")
        type_counts = df['label'].value_counts()
        st.bar_chart(type_counts)
    
    with col2:
        st.subheader("🎯 Confidence Distribution")
        confidence_bins = pd.cut(df['confidence'], bins=10).value_counts().sort_index()
        st.bar_chart(confidence_bins)
    
    st.subheader("📋 All Entities")
    st.dataframe(df[['text', 'label', 'confidence']].sort_values('confidence', ascending=False))

def display_entity_metrics(entities):
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
    if not entities:
        return
    
    by_type = defaultdict(list)
    for entity in entities:
        by_type[entity.label].append(entity)
    
    for entity_type, type_entities in sorted(by_type.items()):
        with st.expander(f"📂 {entity_type} ({len(type_entities)} entities)", expanded=True):
            sorted_entities = sorted(type_entities, key=lambda x: x.confidence, reverse=True)
            
            for entity in sorted_entities[:10]:
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
    return """The latest news on June 24, 2025, is dominated by escalating tensions in the Middle East, primarily between the United States, Israel, and Iran. President Trump has announced that military strikes have "totally obliterated" three Iranian nuclear facilities at Fordow, Natanz, and Isfahan. The Pentagon confirmed that B-52 bombers conducted precision strikes on these facilities. Iran has launched retaliatory missile attacks on a US military base in Qatar, with reports of explosions and damage. Qatar's air defense systems reportedly intercepted some of the incoming missiles. The international community is urging de-escalation, with the United Nations Security Council calling an emergency session. UN Secretary-General expressed "grave alarm" over the situation. Russia and China have warned that continued US attacks on Iran risk triggering a broader global conflict. Oil prices have surged by 15% following the strikes. In other news, the FIFA Club World Cup 2025 continues with FC Porto facing Al Ahly at MetLife Stadium in New Jersey. A nationwide boycott of McDonald's, organized by The People's Union USA, has begun on June 24th, protesting alleged low wages of $12 per hour, tax avoidance of $2.3 billion, and lack of corporate accountability. In the Philippines, Vice President Sara Duterte is facing an impeachment complaint filed by opposition lawmakers, which she has dismissed as politically motivated. Nigeria has been crowned the first-ever African men's and women's flag football champions at the continental championships, while the Oklahoma City Thunder won the NBA championship against the Boston Celtics in a thrilling 7-game series."""

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.markdown('<h1 class="main-header">🧠 Advanced Gemini Parser</h1>', unsafe_allow_html=True)
    st.markdown("**Extract entities from Gemini responses using advanced NLP and pattern matching**")
    
    # System status
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
        
        # Model selection with better descriptions
        available_models = [("Enhanced Pattern Matching (Recommended)", "enhanced-patterns")]
        
        if TRANSFORMERS_AVAILABLE:
            available_models.extend([
                ("BERT Base (High Accuracy)", "bert-base"),
                ("BERT Large (Highest Accuracy, Slow)", "bert-large"),
                ("DistilBERT (Fast & Accurate)", "distilbert")
            ])
        
        if SPACY_AVAILABLE:
            available_models.append(("spaCy NER (Balanced)", "spacy-only"))
        
        model_choice = st.selectbox(
            "🤖 Select NER Model",
            options=available_models,
            format_func=lambda x: x[0],
            index=0,  # Default to enhanced-patterns
            help="Enhanced pattern matching is always available and works reliably"
        )[1]
        
        # Model descriptions
        model_descriptions = {
            "enhanced-patterns": "🔧 Uses comprehensive regex patterns. Works offline, fast processing, good coverage of news entities.",
            "bert-base": "🧠 BERT transformer model. High accuracy, moderate speed, excellent for complex entities.",
            "bert-large": "🎯 Large BERT model. Highest accuracy, slower processing, best for critical analysis.",
            "distilbert": "⚡ Distilled BERT. Good accuracy, fastest transformer option, balanced performance.",
            "spacy-only": "🔍 spaCy NER model. Good accuracy, fast processing, well-balanced general purpose."
        }
        
        st.info(model_descriptions.get(model_choice, ""))
        
        # Initialize parser
        if 'parser' not in st.session_state:
            st.session_state.parser = StreamlitGeminiParser()
        
        if st.button("🚀 Initialize Model", type="primary"):
            success = st.session_state.parser.initialize(model_choice)
            if success:
                st.session_state.model_ready = True
                st.session_state.active_model = st.session_state.parser.model_type
            else:
                st.session_state.model_ready = False
        
        # Status display
        if hasattr(st.session_state, 'model_ready') and st.session_state.model_ready:
            active_model = getattr(st.session_state, 'active_model', 'unknown')
            st.success(f"🟢 {active_model} ready")
        else:
            st.warning("🟡 Click Initialize to start")
        
        # Quick demo
        st.header("🎯 Quick Demo")
        if st.button("📰 Load Sample News"):
            st.session_state.sample_loaded = True
            st.success("Sample loaded! Go to Input tab.")
        
        # Help section
        with st.expander("❓ Need Help?"):
            st.markdown("""
            **How to use:**
            1. Select a model and click Initialize
            2. Choose input method (text, raw data, or file)
            3. Click Extract Entities
            4. View results and download if needed
            
            **Tips:**
            - Enhanced patterns work without internet
            - BERT models give highest accuracy
            - Use sample data to test features
            """)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📝 Input", "🔍 Results", "📊 Analytics"])
    
    with tab1:
        st.header("📝 Input Data")
        
        input_type = st.radio(
            "Choose input method:",
            ["📄 Plain Text", "🔗 Raw Gemini Data", "📁 File Upload"],
            horizontal=True
        )
        
        text_to_process = None
        
        if input_type == "📄 Plain Text":
            default_text = ""
            if hasattr(st.session_state, 'sample_loaded') and st.session_state.sample_loaded:
                default_text = get_sample_data()
            
            text_input = st.text_area(
                "Enter text to analyze:",
                value=default_text,
                height=300,
                placeholder="Enter news articles, social media posts, or any text content to extract entities from...",
                help="Paste any text here. The system will identify people, organizations, locations, dates, and more."
            )
            
            if text_input.strip():
                text_to_process = text_input
        
        elif input_type == "🔗 Raw Gemini Data":
            st.info("💡 **Tip**: Copy raw network response data from browser dev tools when using Gemini")
            
            raw_input = st.text_area(
                "Paste raw Gemini network response:",
                height=250,
                placeholder='[["wrb.fr","hNvQHb","[[[[...your raw response data...]]]]"]]',
                help="Paste the complete raw response from Gemini's network requests"
            )
            
            if raw_input.strip():
                if hasattr(st.session_state, 'model_ready') and st.session_state.model_ready:
                    extracted = st.session_state.parser.extract_text_from_gemini_raw(raw_input)
                    if extracted:
                        st.success(f"✅ Successfully extracted {len(extracted)} characters of text")
                        with st.expander("📄 Extracted Text Preview", expanded=False):
                            preview_text = extracted[:800] + "..." if len(extracted) > 800 else extracted
                            st.write(preview_text)
                        text_to_process = extracted
                    else:
                        st.error("❌ Could not extract readable text from the raw data")
                        st.info("💡 Make sure you've copied the complete network response including the JSON structure")
                else:
                    st.warning("⚠️ Please initialize a model first")
        
        else:  # File Upload
            st.info("📁 **Supported formats**: .txt files containing plain text or raw Gemini data")
            
            uploaded_file = st.file_uploader(
                "Upload a text file:",
                type=['txt'],
                help="Upload a .txt file containing text content or raw Gemini response data"
            )
            
            if uploaded_file:
                try:
                    file_content = uploaded_file.read().decode('utf-8')
                    file_size = len(file_content)
                    st.success(f"✅ File uploaded successfully: **{uploaded_file.name}** ({file_size:,} characters)")
                    
                    # Auto-detect file format
                    if '[["' in file_content or 'wrb.fr' in file_content:
                        st.info("🔍 **Detected**: Raw Gemini data format")
                        if hasattr(st.session_state, 'model_ready') and st.session_state.model_ready:
                            extracted = st.session_state.parser.extract_text_from_gemini_raw(file_content)
                            if extracted:
                                st.success(f"✅ Extracted {len(extracted)} characters from raw data")
                                text_to_process = extracted
                            else:
                                st.error("❌ Could not extract text from the raw data in file")
                        else:
                            st.warning("⚠️ Initialize a model first to process raw data")
                    else:
                        st.info("📝 **Detected**: Plain text format")
                        text_to_process = file_content
                        
                except UnicodeDecodeError:
                    st.error("❌ Could not read file. Please ensure it's a valid UTF-8 text file.")
                except Exception as e:
                    st.error(f"❌ Error reading file: {str(e)}")
        
        # Processing section
        if text_to_process:
            st.markdown("---")
            
            # Text statistics
            word_count = len(text_to_process.split())
            char_count = len(text_to_process)
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.info(f"📊 **Ready to process**: {char_count:,} characters, {word_count:,} words")
            
            with col2:
                # Processing options
                chunk_processing = st.checkbox("Process in chunks", value=char_count > 5000, 
                                             help="Recommended for texts longer than 5000 characters")
            
            with col3:
                if hasattr(st.session_state, 'model_ready') and st.session_state.model_ready:
                    if st.button("🧠 Extract Entities", type="primary", use_container_width=True):
                        with st.spinner("🔍 Extracting entities..."):
                            progress_bar = st.progress(0)
                            
                            try:
                                if chunk_processing and len(text_to_process) > 5000:
                                    # Process in chunks for large texts
                                    chunk_size = 3000
                                    chunks = [text_to_process[i:i+chunk_size] for i in range(0, len(text_to_process), chunk_size)]
                                    all_entities = []
                                    
                                    for i, chunk in enumerate(chunks):
                                        progress_bar.progress((i + 1) / len(chunks))
                                        chunk_entities = st.session_state.parser.extract_entities(chunk)
                                        all_entities.extend(chunk_entities)
                                    
                                    entities = st.session_state.parser._remove_duplicates(all_entities)
                                else:
                                    progress_bar.progress(0.5)
                                    entities = st.session_state.parser.extract_entities(text_to_process)
                                
                                progress_bar.progress(1.0)
                                
                                if entities:
                                    st.session_state.entities = entities
                                    st.session_state.processed_text = text_to_process
                                    st.success(f"🎉 Successfully extracted **{len(entities)}** entities!")
                                    
                                    # Quick stats
                                    entity_types = len(set(e.label for e in entities))
                                    avg_confidence = sum(e.confidence for e in entities) / len(entities)
                                    st.info(f"📈 Found **{entity_types}** different entity types with **{avg_confidence:.1%}** average confidence")
                                    
                                    st.balloons()
                                else:
                                    st.warning("⚠️ No entities found in the text")
                                    st.info("💡 Try using a different model or check if the text contains recognizable entities")
                                    
                            except Exception as e:
                                st.error(f"❌ Processing failed: {str(e)}")
                                st.info("💡 Try using Enhanced Pattern Matching as a fallback")
                                
                            finally:
                                progress_bar.empty()
                else:
                    st.error("❌ Please initialize a model first")
        else:
            st.info("👆 **Choose an input method above** to get started with entity extraction")
    
    with tab2:
        st.header("🔍 Extraction Results")
        
        if hasattr(st.session_state, 'entities') and st.session_state.entities:
            entities = st.session_state.entities
            
            # Display metrics
            display_entity_metrics(entities)
            
            st.markdown("---")
            
            # Entities by type with search
            st.subheader("🏷️ Entities by Type")
            
            # Search/filter options
            col1, col2 = st.columns([3, 1])
            with col1:
                search_term = st.text_input("🔍 Search entities:", placeholder="Type to filter entities...")
            with col2:
                min_confidence = st.slider("Minimum confidence:", 0.0, 1.0, 0.0, 0.05)
            
            # Filter entities
            filtered_entities = entities
            if search_term:
                filtered_entities = [e for e in filtered_entities if search_term.lower() in e.text.lower()]
            if min_confidence > 0:
                filtered_entities = [e for e in filtered_entities if e.confidence >= min_confidence]
            
            if filtered_entities:
                display_entities_by_type(filtered_entities)
                
                if len(filtered_entities) != len(entities):
                    st.info(f"Showing {len(filtered_entities)} of {len(entities)} entities")
            else:
                st.warning("No entities match your search criteria")
            
            # Export section
            st.markdown("---")
            st.header("💾 Export Results")
            
            # Prepare export data
            df_export = pd.DataFrame([e.to_dict() for e in entities])
            
            # Export options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_data = df_export.to_csv(index=False)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                st.download_button(
                    "📥 Download CSV",
                    csv_data,
                    f"gemini_entities_{timestamp}.csv",
                    "text/csv",
                    use_container_width=True,
                    help="Download all entities as a CSV file for Excel/analysis"
                )
            
            with col2:
                json_data = json.dumps([e.to_dict() for e in entities], indent=2)
                st.download_button(
                    "📄 Download JSON",
                    json_data,
                    f"gemini_entities_{timestamp}.json",
                    "application/json",
                    use_container_width=True,
                    help="Download entities as JSON for programming/API use"
                )
            
            with col3:
                # Summary report
                summary_report = f"""# Entity Extraction Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model Used: {getattr(st.session_state, 'active_model', 'Unknown')}

## Summary
- Total Entities: {len(entities)}
- Entity Types: {len(set(e.label for e in entities))}
- Average Confidence: {sum(e.confidence for e in entities) / len(entities):.3f}

## Entities by Type
{chr(10).join([f"- {label}: {len([e for e in entities if e.label == label])}" for label in sorted(set(e.label for e in entities))])}

## Top 10 Entities
{chr(10).join([f"{i+1}. {e.text} ({e.label}) - {e.confidence:.3f}" for i, e in enumerate(sorted(entities, key=lambda x: x.confidence, reverse=True)[:10])])}
"""
                
                st.download_button(
                    "📊 Download Report",
                    summary_report,
                    f"entity_report_{timestamp}.txt",
                    "text/plain",
                    use_container_width=True,
                    help="Download a human-readable summary report"
                )
            
            # Statistics
            st.markdown("---")
            st.subheader("📊 Extraction Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Entity Distribution:**")
                entity_dist = df_export['label'].value_counts()
                for label, count in entity_dist.items():
                    percentage = (count / len(entities)) * 100
                    st.write(f"• **{label}**: {count} entities ({percentage:.1f}%)")
            
            with col2:
                st.write("**Confidence Analysis:**")
                high_conf = len(df_export[df_export['confidence'] > 0.9])
                med_conf = len(df_export[(df_export['confidence'] > 0.7) & (df_export['confidence'] <= 0.9)])
                low_conf = len(df_export[df_export['confidence'] <= 0.7])
                
                st.write(f"• **High confidence (>90%)**: {high_conf} entities")
                st.write(f"• **Medium confidence (70-90%)**: {med_conf} entities")
                st.write(f"• **Lower confidence (≤70%)**: {low_conf} entities")
            
        else:
            st.info("👈 **Process some text first** to see extraction results here!")
            st.markdown("""
            **What you'll see here:**
            - 📊 **Entity metrics** (counts, confidence scores)
            - 🏷️ **Entities grouped by type** (Person, Organization, Location, etc.)
            - 🔍 **Search and filtering** options
            - 💾 **Export capabilities** (CSV, JSON, reports)
            - 📈 **Detailed statistics** and analysis
            """)
    
    with tab3:
        st.header("📊 Analytics & Visualizations")
        
        if hasattr(st.session_state, 'entities') and st.session_state.entities:
            entities = st.session_state.entities
            
            # Analytics summary
            st.subheader("📈 Analysis Summary")
            
            df = pd.DataFrame([e.to_dict() for e in entities])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                most_common_type = df['label'].mode().iloc[0]
                st.metric("🏆 Most Common Type", most_common_type, 
                         f"{len(df[df['label'] == most_common_type])} entities")
            
            with col2:
                highest_conf_entity = df.loc[df['confidence'].idxmax()]
                st.metric("⭐ Highest Confidence", f"{highest_conf_entity['confidence']:.3f}",
                         f"{highest_conf_entity['text'][:15]}...")
            
            with col3:
                model_used = getattr(st.session_state, 'active_model', 'Unknown')
                st.metric("🤖 Model Used", model_used,
                         f"{len(df[df['source_model'] != 'enhanced_patterns'])} AI extracted")
            
            with col4:
                text_length = len(getattr(st.session_state, 'processed_text', ''))
                entities_per_100_chars = (len(entities) / text_length) * 100 if text_length > 0 else 0
                st.metric("📝 Entity Density", f"{entities_per_100_chars:.1f}",
                         "per 100 characters")
            
            # Visualizations
            create_visualizations(entities)
            
            # Advanced analytics
            st.markdown("---")
            st.subheader("🔬 Advanced Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top Entities by Category:**")
                for label in df['label'].unique():
                    label_entities = df[df['label'] == label].nlargest(3, 'confidence')
                    st.write(f"**{label}:**")
                    for _, entity in label_entities.iterrows():
                        st.write(f"  • {entity['text']} ({entity['confidence']:.3f})")
                    st.write("")
            
            with col2:
                st.write("**Processing Insights:**")
                
                # Model performance
                model_counts = df['source_model'].value_counts()
                st.write("**Sources:**")
                for model, count in model_counts.items():
                    percentage = (count / len(entities)) * 100
                    st.write(f"  • {model}: {count} entities ({percentage:.1f}%)")
                
                st.write("")
                st.write("**Quality Metrics:**")
                avg_conf_by_model = df.groupby('source_model')['confidence'].mean()
                for model, avg_conf in avg_conf_by_model.items():
                    st.write(f"  • {model}: {avg_conf:.3f} avg confidence")
        
        else:
            st.info("👈 **Extract entities first** to see analytics and visualizations!")
            st.markdown("""
            **Analytics you'll get:**
            - 📊 **Interactive charts** (pie charts, histograms, scatter plots)
            - 📈 **Entity distribution** analysis
            - 🎯 **Confidence score** breakdowns  
            - 🏆 **Top entities** by category
            - 🤖 **Model performance** comparisons
            - 📝 **Text processing** insights
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        🧠 <strong>Advanced Gemini Parser</strong> | 
        Built with ❤️ using Streamlit & NLP | 
        <a href='https://github.com' target='_blank'>View on GitHub</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
