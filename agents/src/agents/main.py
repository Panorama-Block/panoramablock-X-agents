#!/usr/bin/env python
import os
import sys
import warnings
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import contextmanager
from gridfs import GridFS

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from agents.crew import Agents
import schedule
import pytz
from neo4j import GraphDatabase
import numpy as np
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "twitter_db"
TWEETS_COLLECTION = "tweets"
TWEETS_ZICO_COLLECTION = "tweets_zico"
TWEETS_AVAX_COLLECTION = "tweets_avax"


@contextmanager
def get_mongo_client():
    """
    Context manager for MongoDB client connection
    """
    client = None
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
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


class Neo4jVectorDB:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        self._driver = None
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def connect(self):
        if not self._driver:
            self._driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
        return self._driver

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None

    def ensure_indexes(self):
        """Ensure required indexes exist in Neo4j"""
        with self.connect() as driver:
            with driver.session() as session:
                session.run(
                    """
                CREATE VECTOR INDEX report_vector IF NOT EXISTS
                FOR (r:Report)
                ON r.vector
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 384,
                        `vector.similarity_function`: 'cosine'
                    }
                }
                """
                )

                session.run(
                    """
                CREATE INDEX report_network IF NOT EXISTS
                FOR (r:Report)
                ON r.network
                """
                )

    def text_to_vector(self, text: str):
        """Convert text to vector using SentenceTransformer"""
        return self.model.encode(text)

    def save_report_vector(self, report_text, metadata=None):
        """Save report vector to Neo4j"""
        vector = self.text_to_vector(report_text)
        
        with self.connect() as driver:
            with driver.session() as session:
                result = session.run(
                    """
                    CREATE (r:Report {
                        text: $text,
                        vector: $vector,
                        network: $network,
                        agent_id: $agent_id,
                        report_type: $report_type,
                        created_at: datetime()
                    })
                    RETURN r.created_at as created_at
                    """,
                    text=report_text,
                    vector=vector.tolist(),
                    network=metadata.get("network", "unknown") if metadata else "unknown",
                    agent_id=metadata.get("agent_id", "unknown") if metadata else "unknown",
                    report_type=metadata.get("report_type", "unknown") if metadata else "unknown"
                )
                return result.single()["created_at"]

    def find_similar_reports(self, text, network=None, limit=5, min_score=0.5):
        """
        Find similar reports using vector similarity search

        Args:
            text (str): Text to find similar reports for
            network (str, optional): Filter by specific network
            limit (int): Maximum number of results
            min_score (float): Minimum similarity score (0-1)
        """
        if not text:
            raise ValueError("Search text cannot be empty")

        vector = self.text_to_vector(text)

        with self.connect() as driver:
            with driver.session() as session:
                cypher_query = """
                CALL db.index.vector.queryNodes('report_vector', $limit, $vector)
                YIELD node, score
                WHERE score >= $min_score
                  AND ($network IS NULL OR 
                      ($network IS NOT NULL AND node.network = $network))
                RETURN node.text as text,
                       node.network as network,
                       node.created_at as created_at,
                       node.report_type as report_type,
                       node.metadata as metadata,
                       score
                ORDER BY score DESC
                """

                results = session.run(
                    cypher_query,
                    vector=vector.tolist(),
                    network=network.lower() if network else None,
                    min_score=min_score,
                    limit=limit,
                )
                return list(results)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def save_report_to_vector_db(report_text, metadata=None):
    """
    Save report to vector database
    """
    try:
        vector_db = Neo4jVectorDB()
        vector_db.ensure_indexes()
        created_at = vector_db.save_report_vector(report_text, metadata)
        logger.info(f"Report saved successfully to vector database at {created_at}")
    except Exception as e:
        logger.error(f"Error saving report to vector database: {e}")


def fetch_tweets_from_mongo():
    """
    Fetches tweets from the MongoDB database.
    """
    try:
        with get_mongo_client() as client:
            db = client[DB_NAME]
            collection = db[TWEETS_COLLECTION]

            last_6h = datetime.now() - timedelta(hours=6)
            tweets = list(
                collection.find({"created_at_datetime": {"$gte": last_6h}}).sort(
                    "created_at_datetime", -1
                )
            )

            logger.info(f"Found {len(tweets)} tweets to process")
            return tweets
    except Exception as e:
        logger.error(f"Error fetching tweets: {e}")
        raise


def save_tweet_to_db(tweet, type="zico"):
    """
    Saves a generated tweet to MongoDB with image reference if available
    """
    try:
        with get_mongo_client() as client:
            db = client[DB_NAME]
            collection = db[TWEETS_ZICO_COLLECTION]

            if type == "avax":
                collection = db[TWEETS_AVAX_COLLECTION]

            image_path = "image.png"
            if os.path.exists(image_path):
                image_id = save_image_to_gridfs(image_path)
                tweet["image_id"] = image_id
                os.remove(image_path)
                logger.info("Local image removed after saving to GridFS")

            result = collection.insert_one(tweet)
            logger.info(f"Tweet saved to MongoDB with id: {result.inserted_id}")

    except OperationFailure as e:
        logger.error(f"MongoDB operation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Error saving tweet to MongoDB: {e}")
        raise


def save_image_to_gridfs(image_path: str) -> str:
    """
    Save image to GridFS and return its ID
    """
    try:
        with get_mongo_client() as client:
            db = client[DB_NAME]
            fs = GridFS(db)

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            with open(image_path, "rb") as f:
                file_id = fs.put(f.read(), filename=os.path.basename(image_path))
                logger.info(f"Image saved to GridFS with id: {file_id}")
                return str(file_id)

    except Exception as e:
        logger.error(f"Error saving image to GridFS: {e}")
        raise


def split_tweet_in_parts(tweet: str, header: str = "Zico100x AI here 🤩 this is what leading AI agents said today on X:") -> list[str]:
    """
    Split a tweet into parts based on 'Part X' markers.
    Supports two formats:
    - "Part X (title):" format
    - "Part X:" format
    Excludes the part headers and ensures each part is within character limit.
    Also removes any asterisks from the text.
    """
    import re

    tweet = tweet.replace("*", "")

    part_markers = list(re.finditer(r"Part \d+(?:\s+\([^)]+\))?:", tweet))

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

    for part_idx in range(total_parts):
        part_number = part_idx + 1
        cleaned_section = "\n".join(
            line for line in sections[part_idx].split("\n") if line.strip()
        )

        lines = cleaned_section.split("\n")
        processed_lines = []

        for line_idx, line in enumerate(lines):
            if line_idx == 0 and any(c for c in line[:3] if ord(c) > 127):
                processed_lines.append(line)
                if line_idx + 1 < len(lines) and lines[line_idx + 1].strip():
                    processed_lines.append("")
            else:
                processed_lines.append(line)

        cleaned_section = "\n".join(processed_lines)

        if "#" in cleaned_section:
            lines = cleaned_section.split("\n")
            has_hashtag_line = any(line.strip().startswith("#") for line in lines)

            if not has_hashtag_line:
                hashtag_pattern = r"((?:\s|^)#\w+)+"
                hashtags = re.findall(hashtag_pattern, cleaned_section)

                if hashtags:
                    main_text = re.sub(hashtag_pattern, "", cleaned_section).strip()
                    hashtag_line = " ".join([tag.strip() for tag in hashtags])
                    cleaned_section = f"{main_text}\n\n{hashtag_line}"

        suffix = f" {part_idx}/{total_parts}"
        max_length = 200 - len(suffix)

        if len(cleaned_section) > max_length:
            cut_index = cleaned_section.rfind("\n", 0, max_length)
            if cut_index == -1:
                cut_index = cleaned_section.rfind(". ", 0, max_length)
            if cut_index == -1:
                cut_index = max_length

            cleaned_section = cleaned_section[:cut_index].strip()

        part = f"{cleaned_section.strip()}"
        footer = f"🧵 ({part_number}/{total_parts})"

        if part_idx == 0:
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

        # Clear old images
        cleanup_old_images()

        tweets = fetch_tweets_from_mongo()

        if not tweets:
            logger.warning("No tweets found to process")
            return

        inputs = {"text": "\n".join([tweet["text"] for tweet in tweets])}

        result = Agents().tweet_crew().kickoff(inputs=inputs)

        if hasattr(result, "raw"):
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
            "original_text": tweet_text,
            "parts": tweet_parts,
            "created_at_datetime": datetime.now(),
            "posted": False,
        }
        
        logger.info("Generating image for the tweet")
        image_agent = Agents().image_crew().kickoff(inputs={"text": tweet_parts[0], "type": "zico"})

        save_tweet_to_db(generated_tweet, type="zico")

        logger.info(f"Image generation result: {image_agent}")
        
        # report_path = Path(__file__).resolve().parents[2] / "zico_report.md"
        # if not report_path.exists():
        #     raise FileNotFoundError(f"zico_report.md not found at {report_path}")

        # with open(report_path, "r", encoding="utf-8") as f:
        #     report_text = f.read()
            
        # save_report_to_vector_db(
        #     report_text,
        #     metadata={
        #         "date": datetime.now().isoformat(),
        #         "network": "zico",
        #         "agent_id": "zico_daily_agent",
        #         "report_type": "daily_research",
        #     },
        # )

        logger.info("Daily tweet processing completed successfully")

        return tweet_text, image_agent

    except Exception as e:
        logger.error(f"Error during daily tweet processing: {e}")
        raise


def process_avax_daily_tweets():
    """
    Main function to be executed daily for AVAX research and tweet generation
    """
    try:
        logger.info("Starting daily AVAX tweet processing")

        cleanup_old_images()

        max_retries = 3
        result = None
        
        for attempt in range(max_retries):
            try:
                logger.info(f"AVAX agent attempt {attempt+1}/{max_retries}")
                result = Agents().avax_crew().kickoff()
                
                if result:
                    logger.info("AVAX agent returned a valid result")
                    break
                    
                logger.warning(f"Attempt {attempt+1}: Empty response from AVAX agent")
                time.sleep(5)
            except Exception as e:
                logger.error(f"Error in attempt {attempt+1}: {str(e)}")
                time.sleep(5)
        
        if not result:
            logger.error("All attempts failed, using fallback content")
            tweet_text = "ZicoAvax AI is currently experiencing technical difficulties. We'll be back with AVAX updates soon!"
        else:
            if hasattr(result, "raw"):
                tweet_text = result.raw
            elif isinstance(result, list) and len(result) > 0:
                tweet_text = result[-1].raw
            else:
                tweet_text = str(result)

        tweet_text = tweet_text.strip()
        header = "ZicoAvax AI here 🤩 this is Avalanche (AVAX) news on X:"
        tweet_parts = split_tweet_in_parts(tweet_text, header)

        logger.info(f"Generated AVAX tweet (in {len(tweet_parts)} parts):")
        for part in tweet_parts:
            logger.info(f"Part: {part}")

        generated_tweet = {
            "original_text": tweet_text,
            "parts": tweet_parts,
            "created_at_datetime": datetime.now(),
            "posted": False,
            "type": "avax",
        }
        
        logger.info("Generating image for the AVAX tweet")
        image_agent = Agents().image_crew().kickoff(inputs={"text": tweet_parts[0]})

        logger.info(f"Image generation result: {image_agent}")

        save_tweet_to_db(generated_tweet, type="avax")
        
        # report_path = Path(__file__).resolve().parents[2] / "avax_report.md"
        # if not report_path.exists():
        #     raise FileNotFoundError(f"avax_report.md not found at {report_path}")

        # with open(report_path, "r", encoding="utf-8") as f:
        #     report_text = f.read()
            
        # save_report_to_vector_db(
        #     report_text,
        #     metadata={
        #         "date": datetime.now().isoformat(),
        #         "network": "avax",
        #         "agent_id": "avax_daily_agent",
        #         "report_type": "daily_research",
        #     },
        # )

        logger.info("Daily AVAX tweet processing completed successfully")

        return tweet_text, image_agent

    except Exception as e:
        logger.error(f"Error during AVAX tweet processing: {e}")
        raise


def safely_execute(func):
    """
    Wrapper for safely executing functions
    """
    def wrapper():
        try:
            return func()
        except Exception as e:
            logger.error(f"Error executing {func.__name__}: {e}")
            return None
    return wrapper


def cleanup_old_images():
    """
    Remove images older than 3 days from GridFS
    """
    try:
        with get_mongo_client() as client:
            db = client[DB_NAME]
            fs = GridFS(db)

            three_days_ago = datetime.now() - timedelta(days=3)
            old_files = fs.find({"uploadDate": {"$lt": three_days_ago}})

            count = 0
            for file in old_files:
                fs.delete(file._id)
                count += 1

            if count > 0:
                logger.info(f"Removed {count} images older than 3 days from GridFS")

    except Exception as e:
        logger.error(f"Error cleaning up old images: {e}")
        raise


def should_run_task(scheduled_utc_hour: int) -> bool:
    """
    Checks if the task should run based on the specified UTC hour
    """
    utc_now = datetime.now(pytz.UTC)
    return utc_now.hour == scheduled_utc_hour


def run():
    """
    Configure and run the scheduler
    """

    # Zico
    schedule.every().hour.at(":00").do(
        lambda: should_run_task(6) and process_daily_tweets()
    )
    schedule.every().hour.at(":00").do(
        lambda: should_run_task(12) and process_daily_tweets()
    )
    schedule.every().hour.at(":00").do(
        lambda: should_run_task(18) and process_daily_tweets()
    )
    schedule.every().hour.at(":00").do(
        lambda: should_run_task(22) and process_daily_tweets()
    )
    
    # Avax
    schedule.every().hour.at(":00").do(
        lambda: should_run_task(7) and safely_execute(process_avax_daily_tweets)
    )
    schedule.every().hour.at(":00").do(
        lambda: should_run_task(13) and safely_execute(process_avax_daily_tweets)
    )
    schedule.every().hour.at(":00").do(
        lambda: should_run_task(19) and safely_execute(process_avax_daily_tweets)
    )
    schedule.every().hour.at(":00").do(
        lambda: should_run_task(23) and safely_execute(process_avax_daily_tweets)
    )

    # process_daily_tweets()
    # process_avax_daily_tweets()

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
        Agents().crew().train(
            n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs
        )

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
        Agents().crew().test(
            n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs
        )

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
    

if __name__ == "__main__":
    run()
