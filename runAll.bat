d:\

cd D:\SD\misc\Kafka\kafka_2.12-2.3.0\bin\windows

START zookeeper-server-start.bat ../../config/zookeeper.properties

START kafka-server-start.bat  ../../config/server.properties

START kafka-console-consumer.bat --bootstrap-server localhost:9092 --from-beginning --topic  my-video


