import sqlite3
import pandas as pd
import openai
import json
import time
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# --- 1. SETUP: API KEY AND CLIENT ---
# Import the API key from your separate file
try:
    from Tommyskey import secret_key  # <-- CHANGED
except ImportError:
    print("Error: 'Tommyskey.py' file not found or 'secret_key' not set.") # <-- CHANGED
    print("Please create Tommyskey.py and add: secret_key = 'your_key_here'") # <-- CHANGED
    exit()

# Set up the OpenAI client
try:
    client = openai.OpenAI(api_key=secret_key) # <-- CHANGED
except Exception as e:
    print(f"Error setting up OpenAI client: {e}")
    exit()

print("OpenAI client initialized.")

# --- 2. DATA LOADING ---

def load_data(db_file):
    """Loads reviews from the SQLite database."""
    print(f"Loading data from {db_file}...")
    try:
        conn = sqlite3.connect(db_file)
        # 1. Select the correct column from your database
        df = pd.read_sql_query("SELECT review_text FROM reviews", conn) 
        conn.close()
        
        # 2. Rename the column to 'feedback' so the rest of the script works
        df = df.rename(columns={'review_text': 'feedback'}) 
        
        # 3. Add a unique ID for tracking
        df['review_id'] = range(1, len(df) + 1)
        
        print(f"Successfully loaded {len(df)} reviews.")
        return df
    except Exception as e:
        print(f"Error loading database: {e}")
        print("Please ensure 'feedback.db' exists and the table/column names are correct.")
        return pd.DataFrame() # Return empty DataFrame on error

# --- 3. SENTIMENT ANALYSIS (Objective 1 & Tip A) ---

def get_sentiment(review_text):
    """
    Analyzes the sentiment of a single review using OpenAI API.
    Returns 'Positive', 'Negative', or 'Neutral'.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a sentiment analysis expert. Analyze the customer review and return ONLY one word: Positive, Negative, or Neutral."},
                {"role": "user", "content": f"Review: \"{review_text}\""}
            ],
            temperature=0,
            max_tokens=10
        )
        sentiment = response.choices[0].message.content.strip().capitalize()
        
        # Basic validation
        if sentiment not in ['Positive', 'Negative', 'Neutral']:
            return 'Neutral' # Default to Neutral if output is unexpected
        return sentiment
        
    except Exception as e:
        print(f"Error in sentiment API call: {e}")
        return "Error" # Mark as Error to handle later

# --- 4. ASPECT EXTRACTION (Objective 2 & Tip B) ---

def get_aspects(review_text):
    """
    Extracts specific aspects and their sentiment from a review.
    Returns a JSON string.
    """
    system_prompt = """
    You are an aspect-based sentiment analysis expert. 
    Extract key features/aspects from the review and their associated sentiment (Positive, Negative, or Neutral).
    Return your answer as a JSON list of objects. Each object must have two keys: "aspect" and "sentiment".
    Example: [{"aspect": "screen resolution", "sentiment": "Positive"}, {"aspect": "battery life", "sentiment": "Negative"}]
    If no specific aspects are mentioned, return an empty list [].
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4", # Using GPT-4 for better JSON formatting and accuracy
            response_format={"type": "json_object"}, # Enforce JSON output
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Review: \"{review_text}\""}
            ],
            temperature=0
        )
        # The API in JSON mode often returns a dict, e.g., {"aspects": [...]}. We need to find the list.
        content = response.choices[0].message.content
        data = json.loads(content)
        
        # Find the list within the returned JSON
        if isinstance(data, list):
            return data
        for key, value in data.items():
            if isinstance(value, list):
                return value
        
        return [] # Return empty if no list is found
        
    except Exception as e:
        print(f"Error in aspect API call: {e}")
        return [] # Return empty list on error

# --- 5. DATA PROCESSING (Applying Functions) ---

def process_reviews(df):
    """
    Applies sentiment and aspect analysis to each review.
    Uses for loops as requested.
    """
    print("Starting review processing...")
    
    # a) Apply Sentiment Analysis (Tip A.c)
    sentiments = []
    total = len(df)
    for i, row in df.iterrows():
        print(f"Processing sentiment for review {i+1}/{total}...")
        sentiment = get_sentiment(row['feedback'])
        sentiments.append(sentiment)
        time.sleep(0.5) # Add a small delay to avoid hitting rate limits
        
    df['overall_sentiment'] = sentiments
    print("Overall sentiment analysis complete.")

    # b) Apply Aspect Extraction (Tip B.b)
    aspects_list = []
    for i, row in df.iterrows():
        print(f"Processing aspects for review {i+1}/{total}...")
        aspects = get_aspects(row['feedback'])
        aspects_list.append(aspects)
        time.sleep(0.5) # Add a small delay
        
    df['aspects'] = aspects_list
    print("Aspect extraction complete.")

    # c) Clean and Process Extracted Aspects (Tip B.c, B.d)
    # This "explodes" the DataFrame, creating a new row for each extracted aspect.
    # This links each aspect to its original review's overall sentiment.
    print("Cleaning and normalizing aspect data...")
    aspect_rows = []
    for i, row in df.iterrows():
        for aspect_item in row['aspects']:
            if 'aspect' in aspect_item and 'sentiment' in aspect_item:
                aspect_rows.append({
                    'review_id': row['review_id'],
                    'overall_sentiment': row['overall_sentiment'],
                    'aspect': str(aspect_item['aspect']).lower().strip(), # Normalize
                    'aspect_sentiment': str(aspect_item['sentiment']).capitalize() # Normalize
                })
    
    if not aspect_rows:
        print("Warning: No aspects were extracted from any review.")
        return df, pd.DataFrame() # Return empty aspect DF

    aspects_df = pd.DataFrame(aspect_rows)
    
    # Filter out any malformed data
    aspects_df = aspects_df[aspects_df['aspect_sentiment'].isin(['Positive', 'Negative', 'Neutral'])]
    
    print("Aspect processing complete.")
    return df, aspects_df

# --- 6. ANALYSIS & VISUALIZATION (Objective 3 & Tip C) ---

def analyze_and_visualize(df, aspects_df):
    """Generates and saves all visualizations and analysis."""
    print("Generating analysis and visualizations...")
    
    # a) Sentiment Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='overall_sentiment', data=df, order=['Positive', 'Neutral', 'Negative'], palette='viridis')
    plt.title('Overall Customer Sentiment Distribution', fontsize=16)
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Number of Reviews', fontsize=12)
    plt.savefig('sentiment_distribution.png')
    print("Saved 'sentiment_distribution.png'")

    if aspects_df.empty:
        print("Skipping aspect visualizations as no aspects were found.")
        return

    # b) Aspect Frequency Analysis (Top 15)
    plt.figure(figsize=(12, 8))
    aspects_df['aspect'].value_counts().head(15).sort_values(ascending=True).plot(kind='barh', color='skyblue')
    plt.title('Top 15 Most Frequently Mentioned Aspects', fontsize=16)
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Aspect', fontsize=12)
    plt.tight_layout()
    plt.savefig('aspect_frequency.png')
    print("Saved 'aspect_frequency.png'")

    # c) Aspect Sentiment Analysis
    # Crosstab to see sentiment per aspect
    aspect_sentiment_counts = pd.crosstab(aspects_df['aspect'], aspects_df['aspect_sentiment'])
    
    # Get top 10 most discussed aspects (positive or negative)
    top_aspects = aspects_df['aspect'].value_counts().head(10).index
    top_aspects_sentiment = aspect_sentiment_counts.loc[top_aspects]

    # Stacked bar chart for top aspects
    top_aspects_sentiment.plot(kind='bar', stacked=True, 
                               figsize=(14, 8), 
                               color={'Positive': 'green', 'Neutral': 'grey', 'Negative': 'red'})
    plt.title('Sentiment for Top 10 Mentioned Aspects', fontsize=16)
    plt.xlabel('Aspect', fontsize=12)
    plt.ylabel('Number of Mentions', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Aspect Sentiment')
    plt.tight_layout()
    plt.savefig('top_aspects_sentiment.png')
    print("Saved 'top_aspects_sentiment.png'")

    # d, e, f, g) Get data for recommendations
    # Strengths (Top 5 Positive Aspects)
    positive_aspects = aspects_df[aspects_df['aspect_sentiment'] == 'Positive']['aspect'].value_counts().head(5)
    print("\n--- Key Strengths (Top 5 Positive) ---")
    print(positive_aspects)

    # Weaknesses (Top 5 Negative Aspects)
    negative_aspects = aspects_df[aspects_df['aspect_sentiment'] == 'Negative']['aspect'].value_counts().head(5)
    print("\n--- Key Weaknesses (Top 5 Negative) ---")
    print(negative_aspects)

    # Generate Word Cloud for popular aspects
    all_aspects_text = ' '.join(aspects_df['aspect'])
    if all_aspects_text:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_aspects_text)
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of All Mentioned Aspects', fontsize=16)
        plt.savefig('aspect_wordcloud.png')
        print("Saved 'aspect_wordcloud.png'")

# --- 7. MAIN EXECUTION ---

def main():
    # Load
    reviews_df = load_data('feedback.db')
    if reviews_df.empty:
        return

    # Process
    reviews_df, aspects_df = process_reviews(reviews_df)
    
    # Save processed data to Excel for inspection (optional but recommended)
    try:
        with pd.ExcelWriter('analysis_output.xlsx') as writer:
            reviews_df.to_excel(writer, sheet_name='Overall_Sentiment', index=False)
            if not aspects_df.empty:
                aspects_df.to_excel(writer, sheet_name='Aspect_Sentiment', index=False)
        print("\nSaved all processed data to 'analysis_output.xlsx'")
    except Exception as e:
        print(f"\nCould not save Excel file (is it open?): {e}")


    # Analyze & Visualize
    analyze_and_visualize(reviews_df, aspects_df)
    
    print("\n--- Project Complete ---")
    print("Check the folder for your .png visualization files and 'analysis_output.xlsx'.")

if __name__ == "__main__":
    main()