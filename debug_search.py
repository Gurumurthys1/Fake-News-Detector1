# debug_search.py - Debug search query generation

import requests
import re

def extract_key_terms(headline: str) -> str:
    """Same function as in main.py for testing"""
    # Remove common words that don't help with search
    stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'into', 'by', 'for', 'with', 'at', 'on', 'in', 'to', 'of', 'and', 'or', 'but'}
    
    # Special handling for sports and known entities
    sports_terms = {
        'ipl': 'Indian Premier League IPL',
        'kkr': 'Kolkata Knight Riders KKR',
        'rcb': 'Royal Challengers Bangalore RCB',
        'csk': 'Chennai Super Kings CSK',
        'mi': 'Mumbai Indians MI',
        'dc': 'Delhi Capitals DC',
        'rr': 'Rajasthan Royals RR',
        'pbks': 'Punjab Kings PBKS',
        'gt': 'Gujarat Titans GT',
        'lsg': 'Lucknow Super Giants LSG',
        'srh': 'Sunrisers Hyderabad SRH'
    }
    
    headline_lower = headline.lower()
    
    # Expand sports abbreviations
    expanded_terms = []
    words = re.findall(r'\b[a-zA-Z0-9]+\b', headline.lower())
    
    for word in words:
        if word in sports_terms:
            expanded_terms.extend(sports_terms[word].split())
        elif word not in stop_words and len(word) > 1:
            expanded_terms.append(word)
    
    # Remove duplicates while preserving order
    seen = set()
    key_terms = []
    for term in expanded_terms:
        if term not in seen:
            seen.add(term)
            key_terms.append(term)
    
    # For sports events, use broader terms
    if any(sport in headline_lower for sport in ['ipl', 'cricket', 'champion', 'winner', 'won']):
        # Use OR logic for sports to find more articles
        primary_terms = key_terms[:3]
        secondary_terms = key_terms[3:6] if len(key_terms) > 3 else []
        
        if secondary_terms:
            return f"({' AND '.join(primary_terms)}) OR ({' OR '.join(secondary_terms)})"
        else:
            return ' OR '.join(primary_terms)
    
    # For other news, use the original logic
    if len(key_terms) > 3:
        main_terms = ' '.join(key_terms[:3])
        additional_terms = ' OR '.join(key_terms[3:6])
        return f'"{main_terms}" OR {additional_terms}'
    else:
        return ' OR '.join(key_terms)

def test_search_direct(query, api_key="b8d94d93261a411ca41b535e3e009274"):
    """Test search query directly with NewsAPI"""
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 10,
        "apiKey": api_key,
        "searchIn": "title,description"
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            print(f"Query: '{query}' → Found {len(articles)} articles")
            
            for i, article in enumerate(articles[:3]):
                print(f"  {i+1}. {article['title']} ({article['source']['name']})")
            
            return len(articles)
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return 0
    except Exception as e:
        print(f"Error: {e}")
        return 0

def debug_headline(headline):
    """Debug a specific headline"""
    print(f"\n🔍 Debugging headline: '{headline}'")
    print("=" * 60)
    
    # Test different query strategies
    queries_to_test = [
        ("Original", headline),
        ("Optimized", extract_key_terms(headline)),
        ("Simple IPL", "IPL 2024 winner"),
        ("KKR specific", "Kolkata Knight Riders 2024"),
        ("Simple cricket", "cricket champion 2024"),
        ("Just IPL", "IPL")
    ]
    
    results = []
    for name, query in queries_to_test:
        print(f"\n📊 Testing {name}:")
        count = test_search_direct(query)
        results.append((name, query, count))
    
    print(f"\n📋 Summary for '{headline}':")
    for name, query, count in results:
        print(f"  {name:15} | {count:2d} articles | {query}")

if __name__ == "__main__":
    print("🔍 Search Query Debug Tool")
    print("Testing different query strategies for IPL headlines...")
    
    test_headlines = [
        "KKR won IPL 2024",
        "RCB won IPL 2024", 
        "Kolkata Knight Riders IPL 2024 champions",
        "IPL 2024 winner announced"
    ]
    
    for headline in test_headlines:
        debug_headline(headline)
        print("\n" + "="*80)