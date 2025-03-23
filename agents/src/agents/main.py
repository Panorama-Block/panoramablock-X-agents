#!/usr/bin/env python
import os
import sys
import warnings
import logging
import time
from datetime import datetime, timedelta
from contextlib import contextmanager

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from agents.crew import Agents
import schedule
import pytz

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
DB_NAME = 'twitter_db'
TWEETS_COLLECTION = 'tweets'
TWEETS_ZICO_COLLECTION = 'tweets_zico'

@contextmanager
def get_mongo_client():
    """
    Context manager for MongoDB client connection
    """
    client = None
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB")
        yield client
    except ConnectionFailure as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred with MongoDB: {e}")
        raise
    finally:
        if client:
            client.close()
            logger.debug("MongoDB connection closed")

def fetch_tweets_from_mongo():
    """
    Fetches tweets from the MongoDB database.
    """
    try:
        with get_mongo_client() as client:
            db = client[DB_NAME]
            collection = db[TWEETS_COLLECTION]
            
            last_6h = datetime.now() - timedelta(hours=6)
            tweets = list(collection.find({
                'created_at_datetime': {'$gte': last_6h}
            }).sort('created_at_datetime', -1))
            
            logger.info(f"Found {len(tweets)} tweets to process")
            return tweets
    except Exception as e:
        logger.error(f"Error fetching tweets: {e}")
        raise

def save_tweet_to_db(tweet):
    """
    Saves a generated tweet to MongoDB
    """
    try:
        with get_mongo_client() as client:
            db = client[DB_NAME]
            collection = db[TWEETS_ZICO_COLLECTION]
            
            result = collection.insert_one(tweet)
            logger.info(f"Tweet saved to MongoDB with id: {result.inserted_id}")
            
    except OperationFailure as e:
        logger.error(f"MongoDB operation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Error saving tweet to MongoDB: {e}")
        raise

def split_tweet_in_parts(tweet: str) -> list[str]:
    """
    Split a tweet into parts based on 'Part X' markers.
    Excludes the part headers and ensures each part is within character limit.
    Also removes any asterisks from the text.
    """
    import re

    tweet = tweet.replace('*', '')
    
    part_markers = list(re.finditer(r'Part \d+ \([^)]+\):', tweet))
    
    if not part_markers:
        logger.warning("No part markers found, treating as single part")
        return [f"{tweet.strip()}"]
    
    sections = []
    for i in range(len(part_markers)):
        start = part_markers[i].end()
        
        if i == len(part_markers) - 1:
            content = tweet[start:].strip()
        else:
            end = part_markers[i + 1].start()
            content = tweet[start:end].strip()
            
        sections.append(content)
    
    result = []
    total_parts = len(sections)
    
    header = "Zico100x AI here 🤩 this is what leading AI agents said today on X:"
    
    for i in range(total_parts):
        part_number = i + 1
        cleaned_section = '\n'.join(line for line in sections[i].split('\n') if line.strip())
        
        lines = cleaned_section.split('\n')
        processed_lines = []
        
        for line in lines:
            if ':' in line:
                if len(line) > 3 and any(c for c in line[:3] if ord(c) > 127):
                    processed_lines.append(line)
                else:
                    parts = line.split(':', 1)
                    if len(parts) == 2 and parts[1].strip():
                        processed_lines.append(f"{parts[0].strip()}:\n{parts[1].strip()}")
                    else:
                        processed_lines.append(line)
            else:
                processed_lines.append(line)
        
        cleaned_section = '\n'.join(processed_lines)
        
        suffix = f" {i}/{total_parts}"
        max_length = 200 - len(suffix)
        
        if len(cleaned_section) > max_length:
            cut_index = cleaned_section.rfind('\n', 0, max_length)
            if cut_index == -1:
                cut_index = cleaned_section.rfind('. ', 0, max_length)
            if cut_index == -1:
                cut_index = max_length
            
            cleaned_section = cleaned_section[:cut_index].strip()
        
        part = f"{cleaned_section.strip()}"
        footer = f"🧵 ({part_number}/{total_parts})"
        
        if i == 0:
            formatted_part = f"{header}\n\n{part}\n\n{footer}"
        else:
            formatted_part = f"{part}\n\n{footer}"
            
        result.append(formatted_part)
        
    return result

def process_daily_tweets():
    """
    Main function to be executed daily
    """
    try:
        logger.info("Starting daily tweet processing")
        tweets = fetch_tweets_from_mongo()
        
        if not tweets:
            logger.warning("No tweets found to process")
            return
        
        inputs = {
            'text': "\n".join([tweet['text'] for tweet in tweets])
        }
        
        result = Agents().crew().kickoff(inputs=inputs)
        
        if hasattr(result, 'raw'):
            tweet_text = result.raw
        elif isinstance(result, list) and len(result) > 0:
            tweet_text = result[-1].raw
        else:
            tweet_text = str(result)
        
        tweet_text = tweet_text.strip()
        tweet_parts = split_tweet_in_parts(tweet_text)
        
        logger.info(f"Generated tweet (in {len(tweet_parts)} parts):")
        for part in tweet_parts:
            logger.info(f"Part: {part}")
        
        generated_tweet = {
            'original_text': tweet_text,
            'parts': tweet_parts,
            'created_at_datetime': datetime.now(),
            'posted': False
        }
        save_tweet_to_db(generated_tweet)
        
        logger.info("Daily tweet processing completed successfully")
    except Exception as e:
        logger.error(f"Error during daily tweet processing: {e}")
        raise
    
def should_run_task(scheduled_utc_hour: int) -> bool:
    """
    Verifica se a task deve rodar baseado na hora UTC especificada
    """
    utc_now = datetime.now(pytz.UTC)
    return utc_now.hour == scheduled_utc_hour

def run():
    """
    Configure and run the scheduler
    """
    
    schedule.every().hour.at(":00").do(lambda: should_run_task(6) and process_daily_tweets())
    schedule.every().hour.at(":00").do(lambda: should_run_task(12) and process_daily_tweets())
    schedule.every().hour.at(":00").do(lambda: should_run_task(18) and process_daily_tweets())
    schedule.every().hour.at(":00").do(lambda: should_run_task(22) and process_daily_tweets())
    
    process_daily_tweets()
    
    logger.info("Scheduler iniciado. Aguardando execução...")
    
    while True:
        schedule.run_pending()
        time.sleep(60)

def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "text": "Virtuals' Ecosystem update: Total Market Cap is $1.23B, with a 24h Market Cap Change of -21.92%. The 24h Trading Volume stands at $342.94M. Stay informed, stay ahead! - Vain"
    }
    try:
        Agents().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        Agents().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "text": "Virtuals' Ecosystem update: Total Market Cap is $1.23B, with a 24h Market Cap Change of -21.92%. The 24h Trading Volume stands at $342.94M. Stay informed, stay ahead! - Vain"
    }
    try:
        Agents().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

if __name__ == "__main__":
    run()
