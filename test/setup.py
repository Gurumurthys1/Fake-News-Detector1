# ============================================================================
# SETUP SCRIPT FOR REAL-TIME FAKE NEWS DETECTION SYSTEM
# Run this script first to set up your environment
# ============================================================================

import subprocess
import sys
import os

def install_packages():
    """Install all required packages"""
    print("📦 Installing required packages...")
    
    packages = [
        # Web server packages
        'flask',
        'flask-cors', 
        'flask-socketio',
        
        # Data science packages
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        
        # News collection packages
        'requests',
        'feedparser',
        'beautifulsoup4',
        
        # AI/ML packages
        'google-generativeai',
        'textstat',
        
        # Async packages
        'aiohttp',
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"⚠️ Failed to install {package}")
    
    print("✅ Package installation complete!")

def create_config_file():
    """Create a configuration file for API keys"""
    config_content = '''# ============================================================================
# CONFIGURATION FILE FOR REAL-TIME FAKE NEWS DETECTION SYSTEM
# Replace the placeholder values with your actual API keys
# ============================================================================

# News API Key (Get from: https://newsapi.org/)
NEWSAPI_KEY = "YOUR_NEWSAPI_KEY_HERE"

# Guardian API Key (Get from: https://open-platform.theguardian.com/)
GUARDIAN_API_KEY = "YOUR_GUARDIAN_API_KEY_HERE"

# NY Times API Key (Get from: https://developer.nytimes.com/)
NYTIMES_API_KEY = "YOUR_NYTIMES_API_KEY_HERE"

# Google Gemini API Key (Get from: https://makersuite.google.com/app/apikey)
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"

# ============================================================================
# API KEY SETUP INSTRUCTIONS:
# ============================================================================

"""
1. NewsAPI (Free tier: 1000 requests/day)
   - Go to: https://newsapi.org/
   - Sign up for a free account
   - Copy your API key and replace NEWSAPI_KEY above

2. Guardian API (Free)
   - Go to: https://open-platform.theguardian.com/
   - Register for a developer key
   - Copy your API key and replace GUARDIAN_API_KEY above

3. NY Times API (Free tier: 1000 requests/day)
   - Go to: https://developer.nytimes.com/
   - Create an account and register an app
   - Copy your API key and replace NYTIMES_API_KEY above

4. Google Gemini API (Free tier available)
   - Go to: https://makersuite.google.com/app/apikey
   - Sign in with Google account
   - Create a new API key
   - Copy your API key and replace GEMINI_API_KEY above

Note: You can run the system with just 1-2 API keys configured.
The system will skip sources for which keys are not available.
"""
'''
    
    with open('config.py', 'w') as f:
        f.write(config_content)
    
    print("✅ Created config.py file")
    print("📝 Please edit config.py and add your API keys")

def create_run_script():
    """Create a simple run script"""
    run_script_content = '''#!/usr/bin/env python3
# ============================================================================
# RUN SCRIPT for Real-Time Fake News Detection System
# ============================================================================

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Try to import config
    from config import *
    print("✅ Configuration loaded")
    
    # Update API keys
    API_KEYS = {
        'newsapi': NEWSAPI_KEY,
        'guardian': GUARDIAN_API_KEY, 
        'nytimes': NYTIMES_API_KEY,
        'gemini': GEMINI_API_KEY
    }
    
    # Check which keys are configured
    configured_keys = [k for k, v in API_KEYS.items() if not v.startswith('YOUR_')]
    missing_keys = [k for k, v in API_KEYS.items() if v.startswith('YOUR_')]
    
    print(f"✅ Configured APIs: {configured_keys}")
    if missing_keys:
        print(f"⚠️  Missing APIs: {missing_keys}")
        print("   System will work with limited functionality")
    
except ImportError:
    print("⚠️ config.py not found. Using default configuration.")
    print("   Please run setup.py first to create config.py")
    
    # Default API keys (will have limited functionality)
    API_KEYS = {
        'newsapi': 'YOUR_NEWSAPI_KEY_HERE',
        'guardian': 'YOUR_GUARDIAN_API_KEY_HERE',
        'nytimes': 'YOUR_NYTIMES_API_KEY_HERE',
        'gemini': 'YOUR_GEMINI_API_KEY_HERE'
    }

# Now run the main server
if __name__ == '__main__':
    try:
        # Import and run the server with our API keys
        exec(open('realtime_backend_server.py').read())
    except FileNotFoundError:
        print("❌ realtime_backend_server.py not found!")
        print("   Please make sure all files are in the same directory.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)
'''
    
    with open('run_server.py', 'w') as f:
        f.write(run_script_content)
    
    print("✅ Created run_server.py")

def create_readme():
    """Create a comprehensive README file"""
    readme_content = '''# 🛡️ Real-Time Fake News Detection System

An AI-powered system that monitors news feeds in real-time and detects potentially fake news using machine learning and Google's Gemini AI.

## 🌟 Features

- **Real-Time Monitoring**: Continuously monitors multiple news sources
- **AI-Powered Analysis**: Uses both traditional ML and Google Gemini AI
- **Web Dashboard**: Beautiful, responsive web interface
- **Multiple News Sources**: NewsAPI, Guardian, RSS feeds, and more
- **WebSocket Updates**: Real-time updates without page refresh
- **Custom Article Testing**: Test your own articles for fake news
- **Detailed Analytics**: Comprehensive statistics and charts

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project files
# Run the setup script
python setup.py
```

### 2. Configure API Keys

Edit the `config.py` file and add your API keys:

```python
NEWSAPI_KEY = "your_actual_newsapi_key"
GEMINI_API_KEY = "your_actual_gemini_key"
# ... etc
```

### 3. Run the System

```bash
python run_server.py
```

### 4. Open Your Browser

Navigate to: `http://localhost:5000`

## 🔑 API Keys Setup

### NewsAPI (Free - 1000 requests/day)
1. Go to [https://newsapi.org/](https://newsapi.org/)
2. Sign up for free
3. Copy your API key

### Google Gemini AI (Free tier available)
1. Go to [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
2. Sign in with Google
3. Create new API key

### Guardian API (Free)
1. Go to [https://open-platform.theguardian.com/](https://open-platform.theguardian.com/)
2. Register for developer key

### NY Times API (Optional)
1. Go to [https://developer.nytimes.com/](https://developer.nytimes.com/)
2. Create account and app

## 📊 How It Works

1. **Data Collection**: Monitors multiple news sources via APIs and RSS feeds
2. **ML Analysis**: Traditional machine learning model analyzes text patterns
3. **AI Analysis**: Gemini AI provides detailed credibility assessment
4. **Combined Prediction**: Combines both analyses for final prediction
5. **Real-Time Updates**: Results streamed to dashboard via WebSocket

## 🎛️ Dashboard Features

- **Start/Stop Monitoring**: Control real-time analysis
- **Live Statistics**: Total articles, real vs fake counts, confidence scores
- **Article Feed**: Real-time stream of analyzed articles
- **Custom Testing**: Test your own articles
- **Charts**: Visual trends of analysis results

## 📡 API Endpoints

- `GET /api/status` - System status and statistics
- `POST /api/start` - Start real-time monitoring
- `POST /api/stop` - Stop monitoring
- `POST /api/single_batch` - Analyze single batch
- `POST /api/analyze_custom` - Analyze custom article
- `POST /api/clear_data` - Clear all data

## 🔧 Technical Details

### Backend Components
- **Flask**: Web server and API
- **SocketIO**: Real-time WebSocket communication
- **scikit-learn**: Traditional ML models
- **Google Generative AI**: Gemini AI integration
- **Pandas**: Data processing
- **Threading**: Background monitoring

### Frontend Features
- **Responsive Design**: Works on desktop and mobile
- **Real-Time Updates**: WebSocket-based live updates
- **Interactive Charts**: Chart.js visualizations
- **Modern UI**: Glass-morphism design with animations

### Analysis Pipeline
1. **Text Preprocessing**: Clean and prepare article text
2. **Feature Extraction**: TF-IDF vectorization for ML model
3. **ML Prediction**: Logistic regression classifier
4. **Gemini Analysis**: AI-powered credibility assessment
5. **Result Combination**: Weighted combination of predictions

## 🛠️ Troubleshooting

### Common Issues

1. **"No articles collected"**
   - Check your API keys in config.py
   - Verify internet connection
   - Check API rate limits

2. **"Gemini AI not working"**
   - Ensure Gemini API key is correct
   - Check Google AI Studio for usage limits

3. **"Server won't start"**
   - Install missing packages: `pip install flask flask-socketio`
   - Check port 5000 is available

4. **"Frontend not updating"**
   - Check browser console for errors
   - Verify WebSocket connection
   - Try refreshing the page

### Rate Limits
- NewsAPI: 1000 requests/day (free tier)
- Gemini AI: Check current limits in Google AI Studio
- RSS feeds: Generally no limits

## 📈 Performance Tips

1. **Optimize Monitoring Frequency**: Adjust timing in monitoring loop
2. **Limit Article Batch Size**: Process fewer articles per batch
3. **Cache Results**: Avoid re-analyzing same articles
4. **Use Multiple Sources**: Distribute load across APIs

## 🔒 Security Considerations

- Keep API keys secure and never commit them to version control
- Run on localhost for development
- Use environment variables for production deployment
- Implement rate limiting for production use

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Submit pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review API documentation
3. Check system logs in console
4. Verify all dependencies are installed

## 🔄 Updates

The system automatically fetches the latest news. To update the codebase:
1. Download latest version
2. Run setup.py again if needed
3. Restart the server

---

**Happy Fake News Detection! 🛡️**
'''
    
    with open('README.md', 'w') as f:
        f.write(readme_content)
    
    print("✅ Created README.md")

def create_requirements_file():
    """Create requirements.txt file"""
    requirements = '''# Web Server
flask==2.3.3
flask-cors==4.0.0
flask-socketio==5.3.6

# Data Science
pandas==2.1.1
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2

# News Collection
requests==2.31.0
feedparser==6.0.10
beautifulsoup4==4.12.2

# AI/ML
google-generativeai==0.3.0
textstat==0.7.3

# Async
aiohttp==3.8.6

# Utilities
python-socketio==5.9.0
python-engineio==4.7.1
'''
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("✅ Created requirements.txt")

def main():
    """Main setup function"""
    print("🚀 Setting up Real-Time Fake News Detection System")
    print("=" * 60)
    
    try:
        # Install packages
        install_packages()
        print()
        
        # Create configuration files
        create_config_file()
        create_run_script()
        create_readme()
        create_requirements_file()
        
        print()
        print("✅ Setup Complete!")
        print("=" * 60)
        print("📋 Next Steps:")
        print("1. Edit config.py and add your API keys")
        print("2. Run: python run_server.py")
        print("3. Open: http://localhost:5000")
        print()
        print("📖 Read README.md for detailed instructions")
        print("🔧 Check requirements.txt for package versions")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()