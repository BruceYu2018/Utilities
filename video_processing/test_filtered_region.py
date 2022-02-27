import redis
import time
import json

redis_msg_transfer = redis.StrictRedis(host="localhost", port=6379, db=0)
redis_output_topic = "/test_filterd_reigon"
region = "0, 0, 1080, 1079"
data = dict()
while True:
    time.sleep(0.05)
    timestamp = int(time.time()*1000)
    data["region"] = region
    data["timestamp"] = timestamp
    redis_msg_transfer.set(redis_output_topic, json.dumps(data))