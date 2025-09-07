# ============================================================================
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
