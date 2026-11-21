"""Walking skeleton: Kafka -> Flink -> Kafka, validation only (no model yet)."""
import argparse
import json
import math
import time

from pyflink.common import Types, WatermarkStrategy
from pyflink.common.serialization import SimpleStringSchema
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import (
    DeliveryGuarantee,
    KafkaOffsetsInitializer,
    KafkaRecordSerializationSchema,
    KafkaSink,
    KafkaSource,
)
from pyflink.datastream.functions import MapFunction

BOOTSTRAP = "kafka:19092"  # address of Kafka as seen from other containers
SPEC_PATH = "/opt/ids/models/preprocess_spec.json"


class CheckFlow(MapFunction):
    """Check one flow message and report ok or error (never drop silently)."""

    def open(self, runtime_context):
        with open(SPEC_PATH) as f:
            self.columns = json.load(f)["columns"]

    def map(self, line):
        flow_id = None
        try:
            msg = json.loads(line)
            flow_id = msg.get("flow_id")
            if flow_id is None:
                raise ValueError("missing flow_id")
            features = msg.get("features")
            if not isinstance(features, dict):
                raise ValueError("features is not an object")
            if list(features.keys()) != self.columns:
                raise ValueError("feature names or order differ from the spec")
            for value in features.values():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("non-numeric feature value")
                if not math.isfinite(value):
                    raise ValueError("non-finite feature value")
            return json.dumps({
                "flow_id": flow_id,
                "status": "ok",
                "n_features": len(features),
                "produce_ts_ms": msg.get("produce_ts_ms"),
                "completion_ts_ms": int(time.time() * 1000),
            })
        except Exception as exc:
            return json.dumps({
                "flow_id": flow_id,
                "status": "error",
                "reason": str(exc),
                "completion_ts_ms": int(time.time() * 1000),
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="flows-test")
    parser.add_argument("--output", default="flows-out")
    args = parser.parse_args()

    env = StreamExecutionEnvironment.get_execution_environment()

    source = (
        KafkaSource.builder()
        .set_bootstrap_servers(BOOTSTRAP)
        .set_topics(args.input)
        .set_group_id("ids-skeleton")
        .set_starting_offsets(KafkaOffsetsInitializer.earliest())
        .set_bounded(KafkaOffsetsInitializer.latest())  # stop at current end
        .set_value_only_deserializer(SimpleStringSchema())
        .build()
    )
    sink = (
        KafkaSink.builder()
        .set_bootstrap_servers(BOOTSTRAP)
        .set_record_serializer(
            KafkaRecordSerializationSchema.builder()
            .set_topic(args.output)
            .set_value_serialization_schema(SimpleStringSchema())
            .build()
        )
        .set_delivery_guarantee(DeliveryGuarantee.AT_LEAST_ONCE)
        .build()
    )

    stream = env.from_source(source, WatermarkStrategy.no_watermarks(), "kafka-flows")
    stream.map(CheckFlow(), output_type=Types.STRING()).sink_to(sink)
    env.execute("ids-walking-skeleton")


if __name__ == "__main__":
    main()