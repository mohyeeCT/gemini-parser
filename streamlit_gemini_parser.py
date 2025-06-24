# ============================================================================
# STREAMLIT APP: ADVANCED GEMINI PARSER WITH NER
# ============================================================================
# Beautiful web interface for parsing Gemini responses and extracting entities

import streamlit as st
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import warnings
import io
import base64
from datetime import datetime
warnings.filterwarnings('ignore')

# Try to import NLP libraries (with fallbacks)
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    st.error("⚠️ spaCy not installed. Install with: pip install spacy && python -m spacy download en_core_web_sm")

try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    device = 'cpu'
    st.error("⚠️ Transformers not installed. Install with: pip install transformers torch")

# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================

st.set_page_config(
    page_title="🧠 Advanced Gemini Parser",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #424242;
        border-bottom: 2px solid #E3F2FD;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    .metric-card {
        background: linear-gradient(90deg, #E3F2FD 0%, #BBDEFB 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1E88E5;
    }
    .entity-card {
        background: #F5F5F5;
        padding: 0.8rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
        border-left: 3px solid #4CAF50;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
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

@dataclass
class GeminiResponse:
    original_query: str
    main_text: str
    suggested_queries: List[str]
    links: List[Dict[str, str]]
    entities: List[Entity]
    metadata: Dict[str, Any]

# ============================================================================
# CORE PARSER CLASS
# ============================================================================

@st.cache_resource
class StreamlitGeminiParser:
    """Cached Gemini parser for Streamlit"""
    
    def __init__(self):
        self.ner_pipeline = None
        self.nlp = None
        self.is_initialized = False
        self.model_type = None
    
    def initialize(self, model_choice="bert-base"):
        """Initialize NLP models with caching"""
        
        if self.is_initialized and self.model_type == model_choice:
            return True
        
        models = {
            "bert-base": "dslim/bert-base-NER",
            "bert-large": "dbmdz/bert-large-cased-finetuned-conll03-english",
            "distilbert": "dslim/distilbert-NER",
            "spacy-only": None
        }
        
        try:
            # Load transformer model
            if model_choice != "spacy-only" and TRANSFORMERS_AVAILABLE:
                transformer_model = models[model_choice]
                self.ner_pipeline = pipeline(
                    "ner",
                    model=transformer_model,
                    aggregation_strategy="simple",
                    device=0 if device == 'cuda' else -1
                )
            
            # Load spaCy model
            if SPACY_AVAILABLE:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    st.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
                    return False
            
            self.is_initialized = True
            self.model_type = model_choice
            return True
            
        except Exception as e:
            st.error(f"Error initializing models: {e}")
            return False
    
    def extract_entities(self, text):
        """Extract entities from text"""
        
        if not self.is_initialized:
            return []
        
        entities = []
        
        try:
            # Transformer extraction
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
            
            # spaCy extraction
            if self.nlp:
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
                        confidence=0.95,
                        context=context,
                        source_model="spacy"
                    )
                    entities.append(entity)
            
            # Remove duplicates
            entities = self._remove_duplicates(entities)
            return entities
            
        except Exception as e:
            st.error(f"Error during extraction: {e}")
            return []
    
    def _remove_duplicates(self, entities):
        """Remove overlapping entities"""
        if not entities:
            return []
        
        entities.sort(key=lambda x: x.start_pos)
        unique = []
        
        for entity in entities:
            overlap = False
            for i, existing in enumerate(unique):
                if (entity.start_pos < existing.end_pos and 
                    entity.end_pos > existing.start_pos):
                    if entity.confidence > existing.confidence:
                        unique[i] = entity
                    overlap = True
                    break
            if not overlap:
                unique.append(entity)
        
        return unique
    
    def extract_text_from_gemini_raw(self, raw_data):
        """Extract main text from raw Gemini response"""
        try:
            # Pattern 1: Look for news content
            pattern1 = r'"(The latest news[^"]*(?:\\"[^"]*)*)"'
            match = re.search(pattern1, raw_data, re.IGNORECASE)
            
            if match:
                text = match.group(1)
                text = text.replace('\\"', '"').replace('\\n', '\n')
                return text
            
            # Pattern 2: Look for any long text content
            pattern2 = r'"([^"]{200,})"'
            matches = re.findall(pattern2, raw_data)
            
            if matches:
                longest = max(matches, key=len)
                return longest.replace('\\"', '"').replace('\\n', '\n')
            
            return None
            
        except Exception as e:
            st.error(f"Error extracting text: {e}")
            return None

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_entity_visualizations(entities):
    """Create interactive Plotly visualizations"""
    
    if not entities:
        st.warning("No entities to visualize!")
        return
    
    df = pd.DataFrame([e.to_dict() for e in entities])
    
    # Create tabs for different visualizations
    viz_tabs = st.tabs(["📊 Overview", "🎯 Confidence Analysis", "🏷️ Entity Details", "📈 Advanced Analytics"])
    
    with viz_tabs[0]:  # Overview
        col1, col2 = st.columns(2)
        
        with col1:
            # Entity type distribution
            type_counts = df['label'].value_counts()
            fig_pie = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title="Entity Type Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Model source distribution
            model_counts = df['source_model'].value_counts()
            fig_bar = px.bar(
                x=model_counts.index,
                y=model_counts.values,
                title="Entity Source Models",
                color=model_counts.index,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
    
    with viz_tabs[1]:  # Confidence Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Confidence histogram
            fig_hist = px.histogram(
                df,
                x='confidence',
                bins=20,
                title="Confidence Score Distribution",
                color_discrete_sequence=['#1E88E5']
            )
            fig_hist.update_layout(bargap=0.1)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Box plot by entity type
            fig_box = px.box(
                df,
                x='label',
                y='confidence',
                title="Confidence by Entity Type",
                color='label',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_box.update_xaxes(tickangle=45)
            st.plotly_chart(fig_box, use_container_width=True)
    
    with viz_tabs[2]:  # Entity Details
        # Top entities table
        st.subheader("🏆 Top Entities by Confidence")
        top_entities = df.nlargest(15, 'confidence')[['text', 'label', 'confidence', 'source_model']]
        top_entities['confidence'] = top_entities['confidence'].round(4)
        st.dataframe(
            top_entities,
            use_container_width=True,
            column_config={
                "text": "Entity Text",
                "label": "Type",
                "confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    help="NER confidence score",
                    min_value=0,
                    max_value=1,
                ),
                "source_model": "Model"
            }
        )
    
    with viz_tabs[3]:  # Advanced Analytics
        col1, col2 = st.columns(2)
        
        with col1:
            # Entity length vs confidence scatter
            df['text_length'] = df['text'].str.len()
            fig_scatter = px.scatter(
                df,
                x='confidence',
                y='text_length',
                color='label',
                title="Entity Length vs Confidence",
                hover_data=['text']
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Entity type frequency
            fig_freq = px.bar(
                x=type_counts.values,
                y=type_counts.index,
                orientation='h',
                title="Entity Type Frequency",
                color=type_counts.values,
                color_continuous_scale='Blues'
            )
            fig_freq.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_freq, use_container_width=True)

def display_entity_metrics(entities):
    """Display entity metrics in cards"""
    
    if not entities:
        return
    
    df = pd.DataFrame([e.to_dict() for e in entities])
    
    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="📊 Total Entities",
            value=len(entities)
        )
    
    with col2:
        st.metric(
            label="🎯 Avg Confidence",
            value=f"{df['confidence'].mean():.3f}"
        )
    
    with col3:
        st.metric(
            label="🏷️ Entity Types",
            value=df['label'].nunique()
        )
    
    with col4:
        st.metric(
            label="⭐ High Confidence",
            value=len(df[df['confidence'] > 0.95])
        )
    
    with col5:
        st.metric(
            label="📏 Avg Length",
            value=f"{df['text'].str.len().mean():.1f}"
        )

def display_entities_by_type(entities):
    """Display entities grouped by type"""
    
    if not entities:
        return
    
    # Group entities by type
    by_type = defaultdict(list)
    for entity in entities:
        by_type[entity.label].append(entity)
    
    # Display each type in an expander
    for entity_type, type_entities in sorted(by_type.items()):
        with st.expander(f"📂 {entity_type} ({len(type_entities)} entities)", expanded=False):
            # Sort by confidence
            sorted_entities = sorted(type_entities, key=lambda x: x.confidence, reverse=True)
            
            for entity in sorted_entities:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**{entity.text}**")
                    if entity.context:
                        st.caption(f"Context: {entity.context[:100]}...")
                
                with col2:
                    st.write(f"🎯 {entity.confidence:.3f}")
                
                with col3:
                    st.write(f"🤖 {entity.source_model}")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_download_link(df, filename="entities.csv"):
    """Generate download link for DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download {filename}</a>'
    return href

def process_sample_data():
    """Process sample Gemini data for demo"""
    sample_text = """The latest news on June 24, 2025, is dominated by escalating tensions in the Middle East, primarily between the US, Israel, and Iran. President Trump has stated these strikes "totally obliterated" the Iranian nuclear facilities at Fordow, Natanz, and Isfahan. Iran has launched retaliatory missile attacks on a US base in Qatar, with reports of explosions. Qatar's air defenses reportedly thwarted some of these attacks. The international community is urging de-escalation, with the UN nuclear watchdog expressing "grave alarm." Russia and China have warned that US attacks on Iran risk global conflict. In other news, FIFA Club World Cup 2025 features FC Porto and Al Ahly at MetLife Stadium in New Jersey. A nationwide boycott of McDonald's, organized by The People's Union USA, has begun protesting alleged low wages and tax avoidance."""
    
    return sample_text

# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Advanced Gemini Parser with NER</h1>', unsafe_allow_html=True)
    st.markdown("**Extract entities from Gemini responses using state-of-the-art NLP models**")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_choice = st.selectbox(
            "🤖 Select NER Model",
            options=[
                ("BERT Base (Fast)", "bert-base"),
                ("BERT Large (Accurate)", "bert-large"), 
                ("DistilBERT (Fastest)", "distilbert"),
                ("spaCy Only (Fallback)", "spacy-only")
            ],
            format_func=lambda x: x[0],
            help="Choose the NER model based on your speed/accuracy preference"
        )[1]
        
        # Model info
        model_info = {
            "bert-base": "⚡ Fast processing, good accuracy. Recommended for most use cases.",
            "bert-large": "🎯 Highest accuracy, slower processing. Best for critical analysis.",
            "distilbert": "🚀 Fastest processing, good accuracy. Best for large texts.",
            "spacy-only": "⚡ Very fast, lower accuracy. Fallback option."
        }
        st.info(model_info[model_choice])
        
        # Initialize parser
        st.header("🔧 Model Status")
        
        if 'parser' not in st.session_state:
            st.session_state.parser = StreamlitGeminiParser()
        
        if st.button("🚀 Initialize Models", type="primary"):
            with st.spinner("Loading NLP models..."):
                success = st.session_state.parser.initialize(model_choice)
                if success:
                    st.success("✅ Models loaded successfully!")
                    st.session_state.model_ready = True
                else:
                    st.error("❌ Failed to load models")
                    st.session_state.model_ready = False
        
        # Model status
        if hasattr(st.session_state, 'model_ready') and st.session_state.model_ready:
            st.success("🟢 Models Ready")
        else:
            st.warning("🟡 Initialize models first")
        
        # Sample data button
        st.header("🎯 Quick Demo")
        if st.button("📰 Load Sample News"):
            st.session_state.sample_text = process_sample_data()
            st.success("Sample data loaded!")
    
    # Main content area
    main_tabs = st.tabs(["📝 Input Data", "🔍 Results", "📊 Analytics", "⚙️ Settings"])
    
    with main_tabs[0]:  # Input Data
        st.markdown('<div class="section-header">📝 Input Data</div>', unsafe_allow_html=True)
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Raw Gemini Data", "Plain Text", "Upload File"],
            horizontal=True
        )
        
        text_to_process = None
        
        if input_method == "Raw Gemini Data":
            st.subheader("🔗 Raw Gemini Network Response")
            raw_data = st.text_area(
                "Paste your raw Gemini network response:",
                height=200,
                placeholder='[["wrb.fr","hNvQHb","[[[[...your gemini data...]]]]"]]',
                help="Paste the raw network response data captured from Gemini"
            )
            
            if raw_data.strip():
                if hasattr(st.session_state, 'model_ready') and st.session_state.model_ready:
                    extracted_text = st.session_state.parser.extract_text_from_gemini_raw(raw_data)
                    if extracted_text:
                        st.success(f"✅ Extracted {len(extracted_text)} characters of text")
                        with st.expander("📄 Extracted Text Preview"):
                            st.write(extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text)
                        text_to_process = extracted_text
                    else:
                        st.error("❌ Could not extract text from raw data")
                else:
                    st.warning("⚠️ Please initialize models first")
        
        elif input_method == "Plain Text":
            st.subheader("📝 Plain Text Analysis")
            text_input = st.text_area(
                "Enter text to analyze:",
                height=200,
                placeholder="President Trump announced new sanctions against Iran after the nuclear facility strikes...",
                help="Enter any text for entity extraction"
            )
            
            # Check for sample text in session state
            if hasattr(st.session_state, 'sample_text'):
                if st.button("📰 Use Sample Text"):
                    text_input = st.session_state.sample_text
                    st.rerun()
            
            if text_input.strip():
                text_to_process = text_input
        
        else:  # Upload File
            st.subheader("📁 File Upload")
            uploaded_file = st.file_uploader(
                "Choose a text file",
                type=['txt'],
                help="Upload a .txt file containing text or raw Gemini data"
            )
            
            if uploaded_file:
                file_content = uploaded_file.read().decode('utf-8')
                st.success(f"✅ Uploaded file: {uploaded_file.name} ({len(file_content)} characters)")
                
                # Auto-detect content type
                if '[["' in file_content or 'wrb.fr' in file_content:
                    st.info("🔍 Detected raw Gemini data format")
                    if hasattr(st.session_state, 'model_ready') and st.session_state.model_ready:
                        extracted_text = st.session_state.parser.extract_text_from_gemini_raw(file_content)
                        if extracted_text:
                            text_to_process = extracted_text
                        else:
                            st.error("❌ Could not extract text from file")
                else:
                    st.info("📝 Processing as plain text")
                    text_to_process = file_content
        
        # Process button
        if text_to_process and hasattr(st.session_state, 'model_ready') and st.session_state.model_ready:
            if st.button("🧠 Extract Entities", type="primary", use_container_width=True):
                with st.spinner("🔍 Extracting entities..."):
                    entities = st.session_state.parser.extract_entities(text_to_process)
                    
                    if entities:
                        st.session_state.entities = entities
                        st.session_state.processed_text = text_to_process
                        st.success(f"✅ Extracted {len(entities)} entities!")
                        st.balloons()
                    else:
                        st.warning("⚠️ No entities found in the text")
    
    with main_tabs[1]:  # Results
        st.markdown('<div class="section-header">🔍 Extraction Results</div>', unsafe_allow_html=True)
        
        if hasattr(st.session_state, 'entities') and st.session_state.entities:
            entities = st.session_state.entities
            
            # Display metrics
            display_entity_metrics(entities)
            
            st.markdown("---")
            
            # Display entities by type
            st.subheader("🏷️ Entities by Type")
            display_entities_by_type(entities)
            
            # Download section
            st.markdown("---")
            st.subheader("💾 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Create DataFrame for download
                df_export = pd.DataFrame([e.to_dict() for e in entities])
                
                # Download button
                csv = df_export.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"gemini_entities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # JSON download
                json_data = json.dumps([e.to_dict() for e in entities], indent=2)
                st.download_button(
                    label="📄 Download JSON",
                    data=json_data,
                    file_name=f"gemini_entities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        else:
            st.info("👆 Process some text in the Input Data tab to see results here")
    
    with main_tabs[2]:  # Analytics
        st.markdown('<div class="section-header">📊 Analytics & Visualizations</div>', unsafe_allow_html=True)
        
        if hasattr(st.session_state, 'entities') and st.session_state.entities:
            create_entity_visualizations(st.session_state.entities)
        else:
            st.info("👆 Process some text first to see analytics")
    
    with main_tabs[3]:  # Settings
        st.markdown('<div class="section-header">⚙️ Settings & Information</div>', unsafe_allow_html=True)
        
        # System information
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🖥️ System Information")
            st.info(f"**Device:** {device}")
            st.info(f"**spaCy Available:** {'✅' if SPACY_AVAILABLE else '❌'}")
            st.info(f"**Transformers Available:** {'✅' if TRANSFORMERS_AVAILABLE else '❌'}")
        
        with col2:
            st.subheader("📚 Model Information")
            if hasattr(st.session_state, 'parser') and st.session_state.parser.is_initialized:
                st.success(f"**Active Model:** {st.session_state.parser.model_type}")
                st.success("**Status:** Ready")
            else:
                st.warning("**Status:** Not initialized")
        
        # Installation instructions
        st.subheader("📦 Installation Instructions")
        with st.expander("🔧 Required Dependencies"):
            st.code("""
# Install required packages
pip install streamlit transformers torch spacy pandas matplotlib seaborn plotly

# Download spaCy model
python -m spacy download en_core_web_sm

# Run the app
streamlit run app.py
            """)
        
        # Usage guide
        st.subheader("📖 Usage Guide")
        with st.expander("🚀 How to Use"):
            st.markdown("""
            **Step 1:** Select your preferred NER model in the sidebar
            
            **Step 2:** Click "🚀 Initialize Models" and wait for completion
            
            **Step 3:** Choose your input method:
            - **Raw Gemini Data:** Paste network response data
            - **Plain Text:** Enter any text for analysis  
            - **Upload File:** Upload a .txt file
            
            **Step 4:** Click "🧠 Extract Entities" to process
            
            **Step 5:** View results, analytics, and download data
            """)

# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == "__main__":
    main()

# ============================================================================
# DEPLOYMENT INSTRUCTIONS
# ============================================================================

"""
📦 DEPLOYMENT INSTRUCTIONS:

1. SAVE THIS FILE as 'app.py'

2. CREATE requirements.txt:
   streamlit>=1.28.0
   transformers>=4.21.0
   torch>=1.12.0
   spacy>=3.4.0
   pandas>=1.5.0
   matplotlib>=3.5.0
   seaborn>=0.11.0
   plotly>=5.10.0

3. INSTALL SPACY MODEL:
   python -m spacy download en_core_web_sm

4. RUN LOCALLY:
   streamlit run app.py

5. DEPLOY TO STREAMLIT CLOUD:
   - Push to GitHub
   - Connect to Streamlit Cloud
   - Add packages.txt with: en_core_web_sm https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.4.1/en_core_web_sm-3.4.1.tar.gz

6. ALTERNATIVE DEPLOYMENT (Heroku, etc.):
   - Include Procfile: web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   - Add runtime.txt: python-3.9.16
"""