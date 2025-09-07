# ============================================================================
# REAL-TIME FAKE NEWS DETECTION BACKEND SERVER
# This combines your existing detection system with a Flask web server
# ============================================================================

import os
import sys
import json
import time
import threading
from datetime import datetime, timedelta
from queue import Queue
import asyncio
import logging

# Web server imports
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# Your existing imports (from the previous code)
import pandas as pd
import numpy as np
import requests
import feedparser
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION AND SETUP
# ============================================================================

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# API Keys (Replace with your actual keys)
API_KEYS = {
    'newsapi': 'YOUR_NEWSAPI_KEY_HERE',
    'guardian': 'YOUR_GUARDIAN_API_KEY_HERE',
    'nytimes': 'YOUR_NYTIMES_API_KEY_HERE',
    'gemini': 'YOUR_GEMINI_API_KEY_HERE'
}

# Global variables for real-time processing
is_monitoring = False
monitoring_thread = None
news_queue = Queue()
analysis_stats = {
    'total': 0,
    'real': 0,
    'fake': 0,
    'confidence_sum': 0,
    'start_time': None
}

# ============================================================================
# YOUR EXISTING CLASSES (Modified for real-time use)
# ============================================================================

class GeminiNewsAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = None
        if api_key and 'YOUR_' not in api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-pro')
                logger.info("✅ Gemini model initialized successfully!")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Gemini: {e}")
    
    def analyze_article_with_gemini(self, title, content):
        """Analyze article using Gemini AI for fake news detection"""
        if not self.model:
            return None
        
        prompt = f"""
        Analyze this news article for fake news characteristics. Consider:
        - Sensational language and clickbait elements
        - Lack of credible sources or citations
        - Emotional manipulation tactics
        - Factual inconsistencies or unrealistic claims
        - Writing quality and professionalism
        
        Article Title: {title}
        Article Content: {content[:800]}...
        
        Provide a credibility score (0-100) and brief explanation.
        Respond in JSON format: {{"credibility_score": X, "explanation": "brief analysis"}}
        """
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Try to extract JSON from response
            if '{' in response_text and '}' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_str = response_text[json_start:json_end]
                try:
                    result = json.loads(json_str)
                    return result
                except:
                    pass
            
            # Fallback: parse response manually
            result = {
                'credibility_score': 50,
                'explanation': response_text[:200] if response_text else 'Unable to analyze'
            }
            
            # Try to extract score from text
            import re
            score_match = re.search(r'(\d+)(?:/100|%|\s*(?:out of|score))', response_text)
            if score_match:
                result['credibility_score'] = int(score_match.group(1))
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Gemini analysis error: {e}")
            return {'credibility_score': 50, 'explanation': f'Analysis error: {str(e)}'}

class NewsAPICollector:
    def __init__(self, api_keys):
        self.api_keys = api_keys
    
    def collect_from_newsapi(self, query=None, hours_back=2, max_articles=10):
        """Collect recent articles from NewsAPI"""
        if not self.api_keys.get('newsapi') or 'YOUR_' in self.api_keys['newsapi']:
            logger.warning("⚠️ NewsAPI key not configured")
            return []
            
        url = "https://newsapi.org/v2/everything"
        
        params = {
            'apiKey': self.api_keys['newsapi'],
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': min(max_articles, 50)
        }
        
        if query:
            params['q'] = query
        else:
            from_time = datetime.utcnow() - timedelta(hours=hours_back)
            params['from'] = from_time.isoformat()
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            if data.get('status') == 'ok':
                for article in data.get('articles', []):
                    if article.get('title') and article.get('content'):
                        articles.append({
                            'title': article['title'],
                            'content': (article.get('description', '') + ' ' + 
                                      article.get('content', '')).strip(),
                            'url': article['url'],
                            'source': f"NewsAPI-{article['source']['name']}",
                            'published_at': article['publishedAt'],
                            'reliability_score': 0.7
                        })
            
            logger.info(f"✅ NewsAPI: Collected {len(articles)} articles")
            return articles
            
        except Exception as e:
            logger.error(f"❌ NewsAPI Error: {e}")
            return []
    
    def collect_from_rss_feeds(self, max_per_source=5):
        """Collect from RSS feeds"""
        rss_sources = {
            'BBC News': 'http://feeds.bbci.co.uk/news/rss.xml',
            'Reuters': 'http://feeds.reuters.com/reuters/topNews',
            'CNN': 'http://rss.cnn.com/rss/edition.rss',
        }
        
        all_articles = []
        
        for source_name, rss_url in rss_sources.items():
            try:
                feed = feedparser.parse(rss_url)
                articles = []
                
                for entry in feed.entries[:max_per_source]:
                    content = ''
                    if hasattr(entry, 'content') and entry.content:
                        content = entry.content[0].value
                    elif hasattr(entry, 'description'):
                        content = entry.description
                    elif hasattr(entry, 'summary'):
                        content = entry.summary
                    
                    if entry.title and content:
                        # Clean HTML tags
                        import re
                        content = re.sub(r'<[^>]+>', '', content)
                        
                        articles.append({
                            'title': entry.title,
                            'content': content[:600],
                            'url': entry.link,
                            'source': source_name,
                            'published_at': getattr(entry, 'published', datetime.now().isoformat()),
                            'reliability_score': 0.8
                        })
                
                all_articles.extend(articles)
                logger.info(f"✅ {source_name}: Collected {len(articles)} articles")
                
            except Exception as e:
                logger.error(f"❌ {source_name} Error: {e}")
        
        return all_articles

class RealTimeFakeNewsClassifier:
    def __init__(self, gemini_analyzer):
        self.vectorizer = None
        self.model = None
        self.is_trained = False
        self.gemini_analyzer = gemini_analyzer
        self._train_simple_model()
    
    def _train_simple_model(self):
        """Train with sample data for quick setup"""
        fake_samples = [
            "SHOCKING: You won't believe this incredible secret doctors don't want you to know!",
            "BREAKING: Amazing discovery changes everything! Big pharma HATES this!",
            "INCREDIBLE: This weird trick will shock you! Everyone talking about it!",
            "EXPOSED: Truth they don't want you to see! Share before deleted!"
        ]
        
        real_samples = [
            "Federal Reserve announces interest rate changes following economic indicators.",
            "Scientists publish research findings in peer-reviewed journal Nature.",
            "Local government approves infrastructure budget for upcoming fiscal year.",
            "Technology company reports quarterly earnings exceeding analyst expectations."
        ]
        
        all_texts = fake_samples + real_samples
        labels = [1] * len(fake_samples) + [0] * len(real_samples)  # 1=Fake, 0=Real
        
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf_features = self.vectorizer.fit_transform(all_texts).toarray()
        
        self.model = LogisticRegression(random_state=42)
        self.model.fit(tfidf_features, labels)
        self.is_trained = True
        
        logger.info("✅ ML model trained for real-time analysis")
    
    def predict_article(self, title, content):
        """Enhanced prediction using both ML and Gemini AI"""
        results = {}
        
        # Traditional ML prediction
        if self.is_trained:
            text = f"{title} {content}"
            tfidf_features = self.vectorizer.transform([text]).toarray()
            ml_prediction = self.model.predict(tfidf_features)[0]
            ml_confidence = max(self.model.predict_proba(tfidf_features)[0])
            
            results['ml_prediction'] = 'Fake' if ml_prediction == 1 else 'Real'
            results['ml_confidence'] = ml_confidence * 100
        
        # Gemini AI analysis
        if self.gemini_analyzer.model:
            gemini_result = self.gemini_analyzer.analyze_article_with_gemini(title, content)
            if gemini_result:
                results['gemini_credibility'] = gemini_result.get('credibility_score', 50)
                results['gemini_explanation'] = gemini_result.get('explanation', 'No analysis available')
                results['gemini_prediction'] = 'Real' if gemini_result.get('credibility_score', 50) >= 60 else 'Fake'
        
        # Combined prediction
        if 'ml_prediction' in results and 'gemini_prediction' in results:
            ml_vote = 1 if results['ml_prediction'] == 'Real' else 0
            gemini_vote = 1 if results['gemini_prediction'] == 'Real' else 0
            combined_score = (ml_vote + gemini_vote) / 2
            results['combined_prediction'] = 'Real' if combined_score >= 0.5 else 'Fake'
            results['combined_confidence'] = combined_score * 100
        else:
            results['combined_prediction'] = results.get('ml_prediction', 'Unknown')
            results['combined_confidence'] = results.get('ml_confidence', 50)
        
        return results

# ============================================================================
# INITIALIZE SYSTEM COMPONENTS
# ============================================================================

# Initialize components
gemini_analyzer = GeminiNewsAnalyzer(API_KEYS['gemini'])
news_collector = NewsAPICollector(API_KEYS)
classifier = RealTimeFakeNewsClassifier(gemini_analyzer)

# ============================================================================
# REAL-TIME MONITORING FUNCTIONS
# ============================================================================

def collect_and_analyze_news():
    """Collect and analyze news articles"""
    try:
        # Collect articles from multiple sources
        all_articles = []
        
        # Get from NewsAPI with different queries
        queries = ['breaking news', 'politics', 'technology']
        for query in queries[:1]:  # Limit to avoid rate limits
            articles = news_collector.collect_from_newsapi(query=query, max_articles=3)
            all_articles.extend(articles)
        
        # Get from RSS feeds
        rss_articles = news_collector.collect_from_rss_feeds(max_per_source=2)
        all_articles.extend(rss_articles)
        
        # Analyze each article
        for article in all_articles[:5]:  # Limit to 5 articles per batch
            try:
                analysis_result = classifier.predict_article(article['title'], article['content'])
                
                # Prepare data for frontend
                analyzed_article = {
                    'title': article['title'],
                    'content': article['content'][:300] + '...' if len(article['content']) > 300 else article['content'],
                    'source': article['source'],
                    'url': article['url'],
                    'published_at': article['published_at'],
                    'timestamp': datetime.now().isoformat(),
                    'prediction': analysis_result.get('combined_prediction', 'Unknown'),
                    'ml_confidence': round(analysis_result.get('ml_confidence', 50), 1),
                    'gemini_score': analysis_result.get('gemini_credibility', 50),
                    'gemini_explanation': analysis_result.get('gemini_explanation', 'No analysis available')[:200]
                }
                
                # Add to queue for real-time updates
                news_queue.put(analyzed_article)
                
                # Update statistics
                update_stats(analyzed_article)
                
                # Emit to connected clients via WebSocket
                socketio.emit('new_article', analyzed_article)
                
                time.sleep(2)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error analyzing article: {e}")
        
        logger.info(f"✅ Analyzed batch of {len(all_articles)} articles")
        
    except Exception as e:
        logger.error(f"Error in collect_and_analyze_news: {e}")

def update_stats(article):
    """Update global statistics"""
    global analysis_stats
    
    analysis_stats['total'] += 1
    if article['prediction'] == 'Real':
        analysis_stats['real'] += 1
    else:
        analysis_stats['fake'] += 1
    
    analysis_stats['confidence_sum'] += article['ml_confidence']

def monitoring_loop():
    """Main monitoring loop"""
    global is_monitoring
    
    while is_monitoring:
        try:
            collect_and_analyze_news()
            time.sleep(30)  # Wait 30 seconds between batches
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            time.sleep(10)

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve the main dashboard"""
    return send_from_directory('.', 'realtime_news_dashboard.html')

@app.route('/api/status')
def get_status():
    """Get system status"""
    avg_confidence = (analysis_stats['confidence_sum'] / analysis_stats['total']) if analysis_stats['total'] > 0 else 0
    
    hours_running = 1
    if analysis_stats['start_time']:
        hours_running = max(1, (datetime.now() - analysis_stats['start_time']).total_seconds() / 3600)
    
    return jsonify({
        'is_monitoring': is_monitoring,
        'stats': {
            'total': analysis_stats['total'],
            'real': analysis_stats['real'],
            'fake': analysis_stats['fake'],
            'avg_confidence': round(avg_confidence, 1),
            'articles_per_hour': round(analysis_stats['total'] / hours_running, 1)
        },
        'gemini_status': 'Ready' if gemini_analyzer.model else 'Not Available',
        'last_update': datetime.now().isoformat()
    })

@app.route('/api/start', methods=['POST'])
def start_monitoring():
    """Start real-time monitoring"""
    global is_monitoring, monitoring_thread, analysis_stats
    
    if not is_monitoring:
        is_monitoring = True
        analysis_stats['start_time'] = datetime.now()
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()
        logger.info("🟢 Real-time monitoring started")
        return jsonify({'status': 'started', 'message': 'Real-time monitoring started'})
    
    return jsonify({'status': 'already_running', 'message': 'Monitoring is already running'})

@app.route('/api/stop', methods=['POST'])
def stop_monitoring():
    """Stop real-time monitoring"""
    global is_monitoring
    
    if is_monitoring:
        is_monitoring = False
        logger.info("🔴 Real-time monitoring stopped")
        return jsonify({'status': 'stopped', 'message': 'Real-time monitoring stopped'})
    
    return jsonify({'status': 'already_stopped', 'message': 'Monitoring is already stopped'})

@app.route('/api/single_batch', methods=['POST'])
def fetch_single_batch():
    """Fetch and analyze a single batch of articles"""
    try:
        collect_and_analyze_news()
        return jsonify({'status': 'success', 'message': 'Single batch analyzed'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/analyze_custom', methods=['POST'])
def analyze_custom_article():
    """Analyze a custom article"""
    try:
        data = request.get_json()
        title = data.get('title', '').strip()
        content = data.get('content', '').strip()
        
        if not title or not content:
            return jsonify({'status': 'error', 'message': 'Title and content are required'})
        
        # Analyze the custom article
        analysis_result = classifier.predict_article(title, content)
        
        result = {
            'title': title,
            'content': content[:300] + '...' if len(content) > 300 else content,
            'source': 'Custom Input',
            'url': '#',
            'timestamp': datetime.now().isoformat(),
            'prediction': analysis_result.get('combined_prediction', 'Unknown'),
            'ml_confidence': round(analysis_result.get('ml_confidence', 50), 1),
            'gemini_score': analysis_result.get('gemini_credibility', 50),
            'gemini_explanation': analysis_result.get('gemini_explanation', 'No analysis available')
        }
        
        # Update stats and emit to clients
        update_stats(result)
        socketio.emit('new_article', result)
        
        return jsonify({'status': 'success', 'result': result})
        
    except Exception as e:
        logger.error(f"Error analyzing custom article: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/clear_data', methods=['POST'])
def clear_data():
    """Clear all data and statistics"""
    global analysis_stats
    
    analysis_stats = {
        'total': 0,
        'real': 0,
        'fake': 0,
        'confidence_sum': 0,
        'start_time': None
    }
    
    # Clear the queue
    while not news_queue.empty():
        try:
            news_queue.get_nowait()
        except:
            break
    
    # Emit clear signal to clients
    socketio.emit('clear_data')
    
    return jsonify({'status': 'success', 'message': 'Data cleared'})

# ============================================================================
# SOCKETIO EVENTS
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info("Client connected")
    emit('status', {
        'is_monitoring': is_monitoring,
        'stats': analysis_stats
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info("Client disconnected")

# ============================================================================
# MAIN APPLICATION ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    # Create the HTML file if it doesn't exist
    html_file_path = 'realtime_news_dashboard.html'
    
    if not os.path.exists(html_file_path):
        # Create a simple HTML file that connects to the backend
        html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real-Time Fake News Detection</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body>
    <div style="padding: 20px; font-family: Arial, sans-serif;">
        <h1><i class="fas fa-shield-alt"></i> Real-Time Fake News Detection System</h1>
        <p>Backend server is running. Use the API endpoints or connect via WebSocket for real-time updates.</p>
        
        <div style="margin: 20px 0;">
            <button onclick="startMonitoring()" style="padding: 10px 20px; background: #28a745; color: white; border: none; border-radius: 5px; margin: 5px;">
                <i class="fas fa-play"></i> Start Monitoring
            </button>
            <button onclick="stopMonitoring()" style="padding: 10px 20px; background: #dc3545; color: white; border: none; border-radius: 5px; margin: 5px;">
                <i class="fas fa-stop"></i> Stop Monitoring
            </button>
            <button onclick="fetchBatch()" style="padding: 10px 20px; background: #ffc107; color: black; border: none; border-radius: 5px; margin: 5px;">
                <i class="fas fa-sync"></i> Fetch Single Batch
            </button>
        </div>
        
        <div id="status" style="padding: 10px; background: #f8f9fa; border-radius: 5px; margin: 20px 0;">
            Status: Ready
        </div>
        
        <div id="articles" style="max-height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; border-radius: 5px;">
            <p>No articles analyzed yet. Click "Start Monitoring" to begin.</p>
        </div>
    </div>
    
    <script>
        const socket = io();
        
        socket.on('new_article', function(article) {
            addArticleToFeed(article);
        });
        
        socket.on('clear_data', function() {
            document.getElementById('articles').innerHTML = '<p>Data cleared.</p>';
        });
        
        function addArticleToFeed(article) {
            const articlesDiv = document.getElementById('articles');
            const articleDiv = document.createElement('div');
            articleDiv.style.cssText = 'border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px;';
            
            const predictionColor = article.prediction === 'Real' ? '#28a745' : '#dc3545';
            const predictionIcon = article.prediction === 'Real' ? 'fa-check-circle' : 'fa-exclamation-triangle';
            
            articleDiv.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <h4 style="margin: 0; flex: 1;">${article.title}</h4>
                    <span style="background: ${predictionColor}; color: white; padding: 5px 10px; border-radius: 15px; font-size: 0.8em;">
                        <i class="fas ${predictionIcon}"></i> ${article.prediction}
                    </span>
                </div>
                <p style="color: #666; margin: 5px 0;"><strong>Source:</strong> ${article.source} | <strong>ML:</strong> ${article.ml_confidence}% | <strong>Gemini:</strong> ${article.gemini_score}/100</p>
                <p style="margin: 10px 0;">${article.content}</p>
                <p style="font-size: 0.9em; color: #888; font-style: italic;">${article.gemini_explanation}</p>
                <small style="color: #aaa;">${new Date(article.timestamp).toLocaleString()}</small>
            `;
            
            articlesDiv.insertBefore(articleDiv, articlesDiv.firstChild);
            
            // Keep only last 10 articles
            while (articlesDiv.children.length > 10) {
                articlesDiv.removeChild(articlesDiv.lastChild);
            }
        }
        
        async function startMonitoring() {
            const response = await fetch('/api/start', { method: 'POST' });
            const data = await response.json();
            document.getElementById('status').innerHTML = `Status: ${data.message}`;
        }
        
        async function stopMonitoring() {
            const response = await fetch('/api/stop', { method: 'POST' });
            const data = await response.json();
            document.getElementById('status').innerHTML = `Status: ${data.message}`;
        }
        
        async function fetchBatch() {
            const response = await fetch('/api/single_batch', { method: 'POST' });
            const data = await response.json();
            document.getElementById('status').innerHTML = `Status: ${data.message}`;
        }
        
        // Update status periodically
        setInterval(async () => {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                const statusDiv = document.getElementById('status');
                statusDiv.innerHTML = `
                    Status: ${data.is_monitoring ? '🟢 Running' : '🔴 Stopped'} | 
                    Articles: ${data.stats.total} | 
                    Real: ${data.stats.real} | 
                    Fake: ${data.stats.fake} | 
                    Avg Confidence: ${data.stats.avg_confidence}%
                `;
            } catch (error) {
                console.error('Error fetching status:', error);
            }
        }, 5000);
    </script>
</body>
</html>
        '''
        
        with open(html_file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"✅ Created {html_file_path}")
    
    print("🚀 Starting Real-Time Fake News Detection Server...")
    print("=" * 60)
    print("📝 SETUP INSTRUCTIONS:")
    print("1. Replace API keys in the API_KEYS dictionary with your actual keys")
    print("2. Install required packages: pip install flask flask-cors flask-socketio")
    print("3. Run this script: python realtime_backend_server.py")
    print("4. Open http://localhost:5000 in your browser")
    print("=" * 60)
    print()
    
    # Check API keys
    missing_keys = [k for k, v in API_KEYS.items() if 'YOUR_' in v]
    if missing_keys:
        print(f"⚠️  WARNING: Missing API keys: {missing_keys}")
        print("   The system will work with limited functionality")
    else:
        print("✅ All API keys configured!")
    
    print(f"🌐 Server starting on http://localhost:5000")
    print(f"🔧 API endpoints available at http://localhost:5000/api/")
    print("🔄 Real-time updates via WebSocket")
    
    # Run the Flask app with SocketIO
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)