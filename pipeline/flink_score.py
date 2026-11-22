"""Kafka -> Flink -> Kafka: score every flow with the static ONNX model.

The producer already applied transform() once; Flink must NOT repeat it.
"""
import argparse
import json
import math
import time

import numpy as np
import onnxruntime as ort
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

BOOTSTRAP = "kafka:19092"  # Kafka as seen from other containers
MODELS_DIR = "/opt/ids/models"
MODEL_VERSION = "static-v1"


class ScoreFlow(MapFunction):
    """Validate one flow, score it, and report ok or error."""

    def open(self, runtime_context):
        with open(f"{MODELS_DIR}/preprocess_spec.json") as f:
            self.columns = json.load(f)["columns"]
        with open(f"{MODELS_DIR}/static_ae_meta.json") as f:
            self.threshold = float(json.load(f)["threshold"])
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            f"{MODELS_DIR}/static_ae.onnx",
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

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

            x = np.array(list(features.values()), dtype=np.float32).reshape(1, -1)
            score = float(self.session.run(["score"], {"x": x})[0][0])
            if not math.isfinite(score):
                raise ValueError("model returned a non-finite score")

            return json.dumps({
                "flow_id": flow_id,
                "status": "ok",
                "model_version": MODEL_VERSION,
                "score": score,
                "threshold": self.threshold,
                "is_alert": score > self.threshold,
                "label": msg.get("label"),  # carried for evaluation only
                "produce_ts_ms": msg.get("produce_ts_ms"),
                "completion_ts_ms": int(time.time() * 1000),
            })
        except Exception as exc:
            return json.dumps({
                "flow_id": flow_id,
                "status": "error",
                "reason": str(exc),
                "model_version": MODEL_VERSION,
                "completion_ts_ms": int(time.time() * 1000),
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="flows-test")
    parser.add_argument("--output", default="scores")
    parser.add_argument("--group", default="ids-score")
    parser.add_argument("--unbounded", action="store_true",
                        help="keep running instead of stopping at the current end")
    args = parser.parse_args()

    env = StreamExecutionEnvironment.get_execution_environment()

    builder = (
        KafkaSource.builder()
        .set_bootstrap_servers(BOOTSTRAP)
        .set_topics(args.input)
        .set_group_id(args.group)
        .set_starting_offsets(KafkaOffsetsInitializer.earliest())
        .set_value_only_deserializer(SimpleStringSchema())
    )
    if not args.unbounded:
        builder = builder.set_bounded(KafkaOffsetsInitializer.latest())
    source = builder.build()

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
    stream.map(ScoreFlow(), output_type=Types.STRING()).sink_to(sink)
    env.execute("ids-static-scoring")


if __name__ == "__main__":
    main()