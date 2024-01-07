import csv
import websockets
import json
import asyncio
from websockets.exceptions import ConnectionClosedError
from datetime import datetime, timezone
import time
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS

# FastAPI WebSocket 서버의 주소와 포트 설정
fastapi_server_address = ('localhost', 8000)

# CSV 파일 경로 설정
csv_file_path = 'normal_012_07.csv'

# InfluxDB 설정

influxdb_host = 'http://host:8086'
influxdb_port = 8086
influxdb_database = 'database'
influxdb_measurement = 'measurement'

bucket = "bucket"
org = "croissant"
token = "token"

def str_time_to_timestamp(str_time):
  # Parse the string into a datetime object
    dt = datetime.strptime(str_time, "%Y-%m-%d %H:%M:%S")

    # Convert the datetime to UTC time
    dt_utc = dt.replace(tzinfo=timezone.utc)

    # Calculate the nanoseconds since the Unix epoch
    nanoseconds_since_epoch = int(dt_utc.timestamp() * 1e9)
    
    return nanoseconds_since_epoch


# 배치로 들어가게 작업
async def main():
    try:
        async with websockets.connect("ws://localhost:8000/ws") as websocket:
            # while True:
            with open(csv_file_path, 'r') as csv_file:
              csv_reader = csv.DictReader(csv_file)
              
              input_data = []
              for row in csv_reader:
                  input_data.append(row)  # CSV 파일의 한 줄 데이터

                  if len(input_data) == 64:
                    data_to_send = json.dumps({'message': input_data})
                    await websocket.send(data_to_send)
                    response = await websocket.recv()
                    print(f"Response from server: {response}")
                    response_dict = json.loads(response)

                    root_causes = response_dict['prediction']['root_causes']
                    safety_score = response_dict['prediction']['safety_score']
                    print(f'root_causes: {root_causes}, safety_score: {safety_score}')

                    client = influxdb_client.InfluxDBClient(
                      url=influxdb_host,
                      token=token,
                      org=org
                    )

                    write_api = client.write_api(write_options=SYNCHRONOUS)

                    # Create an InfluxDB point
                    point = influxdb_client.Point("point")
                    point.field("root cause 1", root_causes[0])
                    point.field("root cause 2", root_causes[1])
                    point.field("root cause 3", root_causes[2])
                    # Add multiple fields to the point
                    point.field("safety_score", safety_score)

                    write_api.write(bucket=bucket, org=org, record=point)
                    input_data = input_data[1:]

                    await asyncio.sleep(1)
                  else: # Check if we have reached the last row
                    is_last_row = False
                    try:
                        next_row = next(csv_reader)
                    except StopIteration:
                        is_last_row = True

                    if is_last_row:
                        data_to_send = json.dumps({'message': input_data})
                        await websocket.send(data_to_send)
                        response = await websocket.recv()
                        print(f"Response from server: {response}")
                        
    except ConnectionClosedError as e:
        print(f"Connection closed unexpectedly: {e}")

if __name__ == "__main__":
    asyncio.run(main())