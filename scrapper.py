import requests
import json

def get_articles(total_pages):
    all_articles_data = []
    base_url = "https://www.rtbf.be/article/"

    for i in range(1, total_pages + 1):
        url_json = f"https://bff-service.rtbf.be/oaos/v1.5/pages/en-continu?_page={i}&_limit=100"
        print(f"Fetching articles from {url_json}")
        
        response = requests.get(url_json)
        
        if response.status_code == 200:
            page_json = response.text
            articles = json.loads(page_json)["data"]["articles"]
            
            for article in articles:
                article_data = {
                    "type": article.get("type"),
                    "title": article.get("title"),
                    "summary": article.get("summary"),
                    "topic": article.get("dossierLabel"),
                    "publishedFrom": article.get("publishedFrom"),
                    "majorUpdatedAt": article.get("majorUpdatedAt"),
                    "url": base_url + article["slug"] + "-" + str(article["id"]),  
                }
                all_articles_data.append(article_data)
        else:
            print(f"Failed to fetch page {i}: {response.status_code}")

    return all_articles_data

articles = get_articles(100)

with open("articles.json", "w", encoding='utf-8') as f:
    json.dump(articles, f, ensure_ascii=False, indent=4)

print(f"Fetched {len(articles)} articles.")
