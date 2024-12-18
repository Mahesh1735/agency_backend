from flask import Flask, request, jsonify, g
from langchain_core.messages import HumanMessage, AIMessage, convert_to_openai_messages
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from langgraph.prebuilt import create_react_agent
import os
from dotenv import load_dotenv
from agent import Agent, tools
import json
from flask_cors import CORS
import atexit
from psycopg_pool import PoolTimeout, TooManyRequests
import logging
import time

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()

# Database connection configuration
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
    "keepalives": 1,
    "keepalives_idle": 30,
    "keepalives_interval": 10,
    "keepalives_count": 5
}

# Initialize global variables
pool = None
abot = None
first_request_processed = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_app():
    global pool, abot
    
    # Initialize model
    model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.getenv('OPENAI_API_KEY'))
    
    # Initialize connection pool
    pool = ConnectionPool(
        conninfo=os.getenv('AGENT_STATE_DB_URI'),
        min_size=5,          # Minimum connections to maintain
        max_size=20,         # Maximum connections allowed
        timeout=30,          # Connection acquisition timeout
        max_waiting=100,     # Maximum number of waiting clients
        max_lifetime=3600,   # Maximum connection lifetime in seconds
        kwargs=connection_kwargs,
    )
    
    # Initialize agent
    checkpointer = PostgresSaver(pool)
    abot = Agent(model, tools, checkpointer=checkpointer)

def cleanup_pool():
    global pool
    if pool:
        try:
            pool.close()
        except Exception as e:
            app.logger.error(f"Error closing connection pool: {e}")
        finally:
            pool = None

@atexit.register
def cleanup_at_exit():
    cleanup_pool()

@app.before_request
def before_request():
    global first_request_processed
    if not first_request_processed:
        init_app()
        first_request_processed = True
    
    # Track active connections
    g.request_start_time = time.time()
    
@app.after_request
def after_request(response):
    # Log slow queries
    duration = time.time() - g.request_start_time
    if duration > 1.0:  # Log requests taking more than 1 second
        app.logger.warning(
            f"Slow request: {request.path} took {duration:.2f}s"
        )
    return response

@app.route('/chat', methods=['POST'])
def chat():
    conn = None
    try:
        data = request.get_json()
        query = data.get('query', None)
        thread_id = data.get('thread_id')
        timeout = data.get('timeout', 30)

        # Get connection from pool with timeout
        conn = pool.getconn(timeout=timeout)
        
        # Configure thread ID for state management
        config = {"configurable": {"thread_id": thread_id}}
        
        if not query:
            response = abot.graph.get_state(config).values
        else:
            response = abot.graph.invoke({
                "messages": [HumanMessage(content=query)]
            }, config)

        messages = convert_to_openai_messages(response['messages']) if response.get('messages') else []
        tasks = response.get('tasks', {})

        return jsonify({
            'messages': messages,
            'tasks': tasks
        }), 200

    except PoolTimeout:
        return jsonify({'error': 'Database connection timeout'}), 503
    except TooManyRequests:
        return jsonify({'error': 'Too many concurrent requests'}), 429
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if conn:
            pool.putconn(conn)

@app.route('/update_state', methods=['POST'])
def update_state():
    
    try:
        data = request.get_json()
        query = data.get('query', None)
        tasks = data.get('tasks', None)
        thread_id = data.get('thread_id')
        timeout = data.get('timeout', 30)  # Default 30 second timeout

        # Configure thread ID for state management
        config = {"configurable": {"thread_id": thread_id}}
        state = {}
        if query:
            state['messages'] = [AIMessage(content=query)]
        if tasks:
            state['tasks'] = tasks

        if state:
            abot.graph.update_state(config, state)

        response = abot.graph.get_state(config).values

        # Extract the last message and any tasks
        messages = convert_to_openai_messages(response['messages']) if response.get('messages') else []
        tasks = response.get('tasks', {})
    #     tasks = {'2024-12-12 02:40:53.292651': 
    #     {'id': '2024-12-12 02:40:53.292651',
    #     'type': 'linkedin_post_generator',
    #     'status': 'processing',
    #     'args': 
    #         {'linkedin_page_url': 'https://www.linkedin.com/in/mahesh-kumar-a90000200/',
    #         'post_content': 'hello world',
    #         'target_audience': 'software engineers'},
    #     'results':
    #         [{'id': '2024-12-12 02:40:53.292651_0',
    #         'title': 'hello world',
    #         'body': 'hello world, this is a test post, this can be a long post, taking a lot of lines',
    #         'images_url': ['https://d1.awsstatic.com/s3-pdp-redesign/product-page-diagram_Amazon-S3_HIW%402x.ee85671fe5c9ccc2ee5c5352a769d7b03d7c0f16.png'],
    #         'videos_url': ['https://d1.awsstatic.com/s3-pdp-redesign/product-page-diagram_Amazon-S3_HIW%402x.ee85671fe5c9ccc2ee5c5352a769d7b03d7c0f16.png'],
    #         'documents_url': ['https://d1.awsstatic.com/s3-pdp-redesign/product-page-diagram_Amazon-S3_HIW%402x.ee85671fe5c9ccc2ee5c5352a769d7b03d7c0f16.png'],
    #         'cta': 'https://www.linkedin.com/in/mahesh-kumar-a90000200/'}]},
    # '2024-12-12 02:40:53.292652': 
    #     {'id': '2024-12-12 02:40:53.292652',
    #     'type': 'linkedin_post_generator',
    #     'status': 'processing',
    #     'args': 
    #         {'linkedin_page_url': 'https://www.linkedin.com/in/mahesh-kumar-a90000200/',
    #         'post_content': 'hello world',
    #         'target_audience': 'software engineers'}}}


        return jsonify({
            'messages': messages,
            'tasks': tasks
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health/db', methods=['GET'])
def db_health_check():
    try:
        if not pool:
            return jsonify({'status': 'unhealthy', 'error': 'Connection pool not initialized'}), 503
            
        with pool.connection() as conn:
            conn.execute('SELECT 1')
        
        pool_stats = {
            'size': pool.size,
            'min_size': pool.min_size,
            'max_size': pool.max_size,
            'idle': len([c for c in pool._pool if not c.closed]),
            'busy': pool.size - len([c for c in pool._pool if not c.closed])
        }
        
        return jsonify({
            'status': 'healthy',
            'pool_stats': pool_stats
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
