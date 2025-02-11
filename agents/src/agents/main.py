#!/usr/bin/env python
import sys
import warnings
from datetime import datetime, timedelta
from .agents import Agents
import schedule
import time
from pymongo import MongoClient
import logging

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_mongo_client():
    """
    Cria uma conexão com o MongoDB
    """
    try:
        client = MongoClient('mongodb://localhost:27017/')
        return client
    except Exception as e:
        logger.error(f"Erro ao conectar com MongoDB: {e}")
        raise

def fetch_tweets_from_mongo():
    """
    Busca os tweets do MongoDB das últimas 24 horas
    """
    try:
        client = get_mongo_client()
        db = client['panorama']
        collection = db['tweets']
        
        # Busca tweets das últimas 24 horas
        yesterday = datetime.now() - timedelta(days=1)
        tweets = list(collection.find({
            'created_at': {'$gte': yesterday}
        }))
        
        logger.info(f"Encontrados {len(tweets)} tweets para processar")
        return tweets
    except Exception as e:
        logger.error(f"Erro ao buscar tweets: {e}")
        raise
    finally:
        client.close()

def process_daily_tweets():
    """
    Função principal que será executada diariamente
    """
    try:
        logger.info("Iniciando processamento diário de tweets")
        tweets = fetch_tweets_from_mongo()
        
        if not tweets:
            logger.warning("Nenhum tweet encontrado para processar")
            return
        
        inputs = {
            'texts': [tweet['text'] for tweet in tweets]
        }
        
        Agents().crew().kickoff(inputs=inputs)
        
        logger.info("Processamento diário concluído com sucesso")
    except Exception as e:
        logger.error(f"Erro durante o processamento diário: {e}")

def run_scheduler():
    """
    Configura e executa o scheduler
    """
    schedule.every().day.at("00:00").do(process_daily_tweets)
    
    logger.info("Scheduler iniciado. Aguardando execução...")
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Verifica a cada minuto

def run():
    """
    Run the crew.
    """

    inputs = {
        "text": "Virtuals' Ecosystem update: Total Market Cap is $1.23B, with a 24h Market Cap Change of -21.92%. The 24h Trading Volume stands at $342.94M. Stay informed, stay ahead! - Vain"
    }

    try:
        Agents().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


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
    run_scheduler()
