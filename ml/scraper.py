import pandas as pd
from google_play_scraper import Sort, reviews
import os

apps = {
    'tinder': 'com.tinder',
    'bumble': 'com.bumble.app',
    'hinge': 'co.hinge.app'
}

def scrape_reviews():
    all_reviews = []
    
    for app_name, app_id in apps.items():
        print(f"Scraping {app_name}...")
        try:
            rvs, _ = reviews(
                app_id,
                lang='en',
                country='us',
                sort=Sort.NEWEST,
                count=5000
            )
            for r in rvs:
                r['app_name'] = app_name
            all_reviews.extend(rvs)
            print(f"Scraped {len(rvs)} reviews for {app_name}")
        except Exception as e:
            print(f"Error scraping {app_name}: {e}")

    df = pd.DataFrame(all_reviews)
    
    # Select and rename fields to conform to PRD schema
    df = df.rename(columns={
        'reviewId': 'review_id',
        'content': 'review_text',
        'score': 'star_rating',
        'at': 'review_date',
        'thumbsUpCount': 'thumbs_up'
    })
    
    columns_to_keep = ['review_id', 'app_name', 'review_text', 'star_rating', 'review_date', 'thumbs_up']
    
    for col in columns_to_keep:
        if col not in df.columns:
            df[col] = None
            
    df = df[columns_to_keep]
    
    os.makedirs('data', exist_ok=True)
    out_path = 'data/raw_reviews.csv'
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} total reviews to {out_path}")

if __name__ == "__main__":
    scrape_reviews()
