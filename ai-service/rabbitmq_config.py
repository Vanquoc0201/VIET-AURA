# rabbitmq_config.py
import pika

def get_connection():
    return pika.BlockingConnection(pika.ConnectionParameters("localhost"))

def declare_queues(channel):
    # Khởi tạo exchange và queue
    channel.exchange_declare(exchange='request_exchange', exchange_type='direct')
    channel.exchange_declare(exchange='response_exchange', exchange_type='direct')

    channel.queue_declare(queue='image_predict_queue')
    channel.queue_bind(exchange='request_exchange', queue='image_predict_queue', routing_key='image.predict')

    channel.queue_declare(queue='prediction_result_queue')
