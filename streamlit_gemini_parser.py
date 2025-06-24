# ============================================================================
# FILE: requirements.txt
# ============================================================================

# Core dependencies (always required)
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0

# NLP libraries (with fallbacks)
transformers>=4.21.0
torch>=1.12.0
spacy>=3.4.0

# Visualization (optional)
plotly>=5.10.0

# Additional utilities
scikit-learn>=1.1.0

# ============================================================================
# FILE: packages.txt
# ============================================================================

# System packages for spaCy model download
python3-dev
build-essential

# ============================================================================
# FILE: .streamlit/config.toml
# ============================================================================

[theme]
primaryColor = "#1E88E5"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

# ============================================================================
# FILE: setup.sh (Optional - for advanced setup)
# ============================================================================

#!/bin/bash

# Download spaCy model
python -m spacy download en_core_web_sm

echo "Setup complete!"

# ============================================================================
# GITHUB REPOSITORY STRUCTURE
# ============================================================================


# ============================================================================
# FILE: README.md
# ============================================================================

# 🧠 Advanced Gemini Parser with NER

Extract entities from Gemini responses using state-of-the-art NLP models and pattern matching.

## 🚀 Features

- **Multiple NER Models**: BERT, DistilBERT, spaCy, and fallback pattern matching
- **Smart Fallbacks**: Always works even if advanced models fail
- **Interactive Visualizations**: Plotly charts and analytics
- **Multiple Input Methods**: Raw Gemini data, plain text, file upload
- **Export Capabilities**: Download results as CSV or JSON
- **Real-time Processing**: Instant entity extraction and analysis

## 🎯 Live Demo

**[Try it now on Streamlit Cloud!](https://your-app-name.streamlit.app)**

## 📋 Usage

1. **Choose Model**: Select from BERT, spaCy, or basic pattern matching
2. **Initialize**: Click "Initialize Model" and wait for setup
3. **Input Data**: 
   - Paste raw Gemini network response data
   - Enter plain text for analysis
   - Upload .txt files
4. **Extract**: Click "Extract Entities" to process
5. **Analyze**: View results, charts, and export data

## 🛠️ Local Development

```bash
# Clone repository
git clone https://github.com/yourusername/gemini-parser.git
cd gemini-parser

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run locally
streamlit run app.py
```

## 📦 Dependencies

- **Core**: Streamlit, pandas, numpy
- **NLP**: transformers, torch, spaCy
- **Visualization**: plotly
- **Fallbacks**: Built-in pattern matching always available

## 🔧 Model Options

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| BERT Base | Medium | High | Recommended for most users |
| BERT Large | Slow | Highest | Maximum accuracy needed |
| DistilBERT | Fast | Good | Large texts, speed priority |
| spaCy | Fast | Good | Balanced performance |
| Pattern Matching | Fastest | Basic | Always available fallback |

## 📊 Supported Entity Types

- **PERSON**: Names of people, titles + names
- **ORGANIZATION**: Companies, institutions, groups
- **LOCATION**: Countries, cities, regions, facilities
- **DATE**: Dates, times, temporal expressions
- **EVENT**: Conferences, competitions, incidents
- **MONEY**: Currency amounts and financial values
- **FACILITY**: Nuclear facilities, military bases, stadiums

## 🌟 Examples

### Input:
```
President Trump announced strikes on Iranian nuclear facilities at Fordow and Natanz. 
The UN Security Council will meet tomorrow to discuss the Middle East crisis.
```

### Output:
- **PERSON**: President Trump (conf: 0.998)
- **LOCATION**: Iranian, Fordow, Natanz, Middle East (conf: 0.995+)
- **ORGANIZATION**: UN Security Council (conf: 0.992)
- **DATE**: tomorrow (conf: 0.889)
- **FACILITY**: nuclear facilities (conf: 0.887)

## 🚀 Deployment

### Streamlit Cloud (Recommended)

1. Fork this repository
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy automatically with all fallbacks

### Other Platforms

- **Heroku**: Add `Procfile` with `web: streamlit run app.py --server.port=$PORT`
- **Railway**: Works out of the box
- **Replit**: Import repository and run

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙋‍♂️ Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: your-email@example.com

---

Made with ❤️ using Streamlit and advanced NLP models.

# ============================================================================
# FILE: .gitignore
# ============================================================================

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyTorch
*.pth
*.pt

# Transformers cache
.transformers_cache/
transformers_cache/

# spaCy models
*.gz

# Jupyter Notebook
.ipynb_checkpoints

# Environment variables
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Streamlit
.streamlit/secrets.toml

# Data files
*.csv
*.json
*.txt
!requirements.txt
!packages.txt

# ============================================================================
# DEPLOYMENT STEPS FOR STREAMLIT CLOUD
# ============================================================================

## Step 1: Create GitHub Repository

1. Create new repository on GitHub
2. Upload these files:
   - app.py
   - requirements.txt
   - packages.txt
   - .streamlit/config.toml
   - README.md
   - .gitignore

## Step 2: Deploy to Streamlit Cloud

1. Go to https://streamlit.io/cloud
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Choose main branch
6. Set main file path: `app.py`
7. Click "Deploy!"

## Step 3: Configuration (Optional)

### Advanced Settings:
- **Python version**: 3.9 or 3.10
- **Secrets**: Add any API keys if needed
- **Resources**: Default is fine for this app

### Custom Domain (Optional):
- Go to app settings
- Add custom domain
- Update DNS records

## Step 4: Monitor Deployment

1. Watch build logs for any errors
2. App will be available at: `https://your-app-name.streamlit.app`
3. Test all features:
   - Model initialization
   - Text processing
   - Visualizations
   - File downloads

## Step 5: Troubleshooting

### Common Issues:

**1. spaCy Model Download Fails:**
```bash
# Add to packages.txt:
python3-dev
build-essential

# Or modify app.py to handle gracefully
```

**2. Transformers Models Too Large:**
```python
# Use smaller models in production:
"distilbert-base-uncased" instead of "bert-large"
```

**3. Memory Limits:**
```python
# Force CPU usage:
device = -1  # Always use CPU

# Reduce batch size:
chunk_size = 1000  # Process in smaller chunks
```

**4. Slow Loading:**
```python
# Use @st.cache_resource for model loading
# Implement lazy loading
# Show progress indicators
```

## 🎯 Production Optimizations

### For Better Performance:

1. **Model Caching**: Use `@st.cache_resource`
2. **Lazy Loading**: Load models only when needed
3. **CPU Optimization**: Force CPU usage for cloud deployment
4. **Chunking**: Process large texts in chunks
5. **Fallbacks**: Always provide working alternatives

### For Better UX:

1. **Progress Bars**: Show loading progress
2. **Error Handling**: Graceful degradation
3. **Sample Data**: Provide working examples
4. **Clear Instructions**: Help users get started
5. **Mobile Friendly**: Responsive design

---

## 🚀 Your app is now ready for deployment!

Simply create the GitHub repository with these files and connect to Streamlit Cloud. The app will work even if some packages fail to install, thanks to the built-in fallbacks.
