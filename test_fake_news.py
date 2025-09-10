# test_fake_news.py - Test script for the fake news detection system

import requests
import json
import time
from typing import List, Dict

# Test configuration
BASE_URL = "http://127.0.0.1:8000"
TEST_HEADLINES = [
    # Real news examples
    {
        "headline": "NASA successfully launches James Webb Space Telescope",
        "expected": "real",
        "category": "science"
    },
    {
        "headline": "President Biden signs infrastructure bill into law",
        "expected": "real", 
        "category": "politics"
    },
    {
        "headline": "Apple announces new iPhone with improved camera",
        "expected": "real",
        "category": "technology"
    },
    
    # Fake news examples
    {
        "headline": "Aliens confirmed to have landed in Times Square by NASA",
        "expected": "fake",
        "category": "conspiracy"
    },
    {
        "headline": "Scientists discover cure for aging, people can now live 500 years",
        "expected": "fake",
        "category": "medical"
    },
    {
        "headline": "Earth proven to be flat by new satellite images",
        "expected": "fake",
        "category": "conspiracy"
    },
    
    # Uncertain examples
    {
        "headline": "Local school board discusses new lunch menu options",
        "expected": "uncertain",
        "category": "local"
    },
    {
        "headline": "Mysterious lights seen over rural town spark UFO theories",
        "expected": "uncertain",
        "category": "mystery"
    }
]

def test_health_endpoint():
    """Test the health endpoint"""
    print("🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            print(f"   Services: {data['services']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

def test_single_headline(headline: str, expected: str = None, category: str = None) -> Dict:
    """Test a single headline"""
    print(f"\n🔍 Testing: '{headline}'")
    print(f"   Category: {category}, Expected: {expected}")
    
    try:
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/check-news",
            json={"headline": headline},
            timeout=30
        )
        
        end_time = time.time()
        response_time = round((end_time - start_time) * 1000, 2)
        
        if response.status_code == 200:
            data = response.json()
            verdict = data.get("verdict", "Unknown").lower()
            confidence = data.get("confidence", 0)
            sources_checked = data.get("sources_checked", 0)
            reasoning = data.get("reasoning", "No reasoning provided")
            
            # Check if prediction matches expectation
            is_correct = expected is None or verdict == expected.lower()
            status_emoji = "✅" if is_correct else "⚠️"
            
            print(f"{status_emoji} Result: {verdict.upper()} ({confidence}% confidence)")
            print(f"   Sources checked: {sources_checked}")
            print(f"   Response time: {response_time}ms")
            print(f"   Reasoning: {reasoning[:100]}...")
            
            return {
                "headline": headline,
                "verdict": verdict,
                "confidence": confidence,
                "expected": expected,
                "correct": is_correct,
                "response_time": response_time,
                "sources_checked": sources_checked,
                "category": category
            }
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return {
                "headline": headline,
                "error": f"HTTP {response.status_code}",
                "category": category
            }
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {
            "headline": headline,
            "error": str(e),
            "category": category
        }

def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("🚀 Starting Fake News Detector Test Suite")
    print("=" * 50)
    
    # Test health endpoint first
    if not test_health_endpoint():
        print("❌ Server health check failed. Make sure the server is running.")
        return
    
    print(f"\n🧪 Testing {len(TEST_HEADLINES)} headlines...")
    
    results = []
    for i, test_case in enumerate(TEST_HEADLINES, 1):
        print(f"\n[{i}/{len(TEST_HEADLINES)}]", end=" ")
        result = test_single_headline(
            test_case["headline"],
            test_case["expected"],
            test_case["category"]
        )
        results.append(result)
        
        # Small delay between requests
        time.sleep(1)
    
    # Generate test report
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    total_tests = len(results)
    successful_tests = len([r for r in results if "error" not in r])
    correct_predictions = len([r for r in results if r.get("correct", False)])
    
    print(f"Total tests: {total_tests}")
    print(f"Successful requests: {successful_tests}")
    print(f"Correct predictions: {correct_predictions}")
    
    if successful_tests > 0:
        accuracy = (correct_predictions / successful_tests) * 100
        print(f"Accuracy: {accuracy:.1f}%")
        
        # Average response time
        response_times = [r.get("response_time", 0) for r in results if "response_time" in r]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        print(f"Average response time: {avg_response_time:.1f}ms")
        
        # Confidence analysis
        confidences = [r.get("confidence", 0) for r in results if "confidence" in r]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        print(f"Average confidence: {avg_confidence:.1f}%")
    
    # Detailed results by category
    print(f"\n📋 DETAILED RESULTS BY CATEGORY")
    categories = set(r.get("category", "unknown") for r in results)
    
    for category in sorted(categories):
        category_results = [r for r in results if r.get("category") == category]
        category_correct = len([r for r in category_results if r.get("correct", False)])
        
        print(f"\n{category.upper()}:")
        for result in category_results:
            if "error" in result:
                print(f"  ❌ {result['headline'][:50]}... - ERROR: {result['error']}")
            else:
                verdict = result.get("verdict", "unknown").upper()
                confidence = result.get("confidence", 0)
                correct = "✅" if result.get("correct", False) else "❌"
                print(f"  {correct} {result['headline'][:50]}... - {verdict} ({confidence}%)")
    
    print(f"\n🎯 Test completed! Check the results above for detailed analysis.")

def test_edge_cases():
    """Test edge cases and error conditions"""
    print("\n🔬 Testing edge cases...")
    
    edge_cases = [
        "",  # Empty string
        "a",  # Too short
        "This is a very long headline that exceeds normal length limits and contains lots of words to test how the system handles extremely long input text that might cause issues with API limits or processing capabilities" * 3,  # Very long
        "🚀🔥💯 Emoji headline test 🌟⚡🎉",  # Emojis
        "BREAKING: ALL CAPS HEADLINE TEST",  # All caps
        "Test headline with special chars: @#$%^&*()",  # Special characters
    ]
    
    for case in edge_cases:
        print(f"\n🧪 Edge case: '{case[:50]}{'...' if len(case) > 50 else ''}'")
        result = test_single_headline(case)
        if "error" in result:
            print(f"   Expected error for edge case")

if __name__ == "__main__":
    print("🎯 Fake News Detector Test Suite v2.0")
    print("Make sure your server is running at http://127.0.0.1:8000")
    
    choice = input("\nSelect test type:\n1. Comprehensive test\n2. Edge cases only\n3. Both\nEnter choice (1-3): ")
    
    if choice in ["1", "3"]:
        run_comprehensive_test()
    
    if choice in ["2", "3"]:
        test_edge_cases()
    
    print("\n🏁 Testing completed!")