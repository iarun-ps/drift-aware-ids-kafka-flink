FROM flink:1.20.5-java11

# Python for the job (container's own Python)
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends python3 python3-pip python3-dev wget && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3 /usr/bin/python

# PyFlink (same version as Flink) and ONNX Runtime
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --default-timeout=300 --retries 20 \
    apache-flink==1.20.5 onnxruntime

# Kafka connector for Flink 1.20
RUN wget -q -P /opt/flink/lib/ \
    https://repo.maven.apache.org/maven2/org/apache/flink/flink-sql-connector-kafka/3.4.0-1.20/flink-sql-connector-kafka-3.4.0-1.20.jar && \
    chown flink:flink /opt/flink/lib/flink-sql-connector-kafka-3.4.0-1.20.jar