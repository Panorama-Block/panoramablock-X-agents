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
        # Verify connection
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
            
            yesterday = datetime.now() - timedelta(days=1)
            tweets = list(collection.find({
                'created_at_datetime': {'$gte': yesterday}
            }).sort('created_at_datetime', -1))[0:10]  # Sort by newest first
            
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
            
            tweet_data = {
                'text': tweet['text'],
                'created_at_datetime': tweet['created_at_datetime']
            }
            
            result = collection.update_one(
                {
                    'text': tweet['text'],
                    'created_at_datetime': tweet['created_at_datetime']
                },
                {'$set': tweet_data},
                upsert=True
            )
            
            if result.upserted_id:
                logger.info(f"New tweet saved to MongoDB with id: {result.upserted_id}")
            else:
                logger.info(f"Tweet updated in MongoDB. Modified count: {result.modified_count}")
            
    except OperationFailure as e:
        logger.error(f"MongoDB operation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Error saving tweet to MongoDB: {e}")
        raise

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
        
        logger.info(f"Generated tweet: {tweet_text}")
        
        generated_tweet = {
            'text': tweet_text,
            'created_at_datetime': datetime.now()
        }
        save_tweet_to_db(generated_tweet)
        
        logger.info("Daily tweet processing completed successfully")
    except Exception as e:
        logger.error(f"Error during daily tweet processing: {e}")
        raise

def run():
    """
    Configure and run the scheduler
    """
    # schedule.every().day.at("00:00").do(process_daily_tweets)
    # every hour
    process_daily_tweets()
    schedule.every().hour.do(process_daily_tweets)
    
    logger.info("Scheduler iniciado. Aguardando execução...")
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Verifica a cada minuto

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
