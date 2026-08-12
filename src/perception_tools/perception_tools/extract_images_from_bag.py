import argparse
from pathlib import Path

import cv2
import numpy as np

import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image

SUPPORTED_ENCODINGS = {
    "rgb8",
    "bgr8",
    "rgba8",
    "bgra8",
    "mono8",
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract PNG images from a ROS 2 bag Image topic."
    )
    parser.add_argument(
        "--bag",
        required=True,
        help="Path to the ROS 2 bag directory, for example bags/terrain_rgb_01.",
    )
    parser.add_argument(
        "--topic",
        default="/rgbd_camera/image",
        help="Image topic to extract. Default: /rgbd_camera/image.",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/raw_images",
        help="Output directory for PNG files. Default: dataset/raw_images.",
    )
    parser.add_argument(
        "--every-nth",
        type=int,
        default=1,
        help="Save every Nth image message from the selected topic. Default: 1.",
    )
    parser.add_argument(
        "--min-interval",
        type=float,
        default=0.0,
        help="Minimum time in seconds between saved images. Default: 0.0.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Stop after saving this many images. 0 means no limit.",
    )
    parser.add_argument(
        "--session-name",
        default=None,
        help="Name used in output filenames, for example terrain_rgb_03."
    )
    parser.add_argument(
        "--start-index",
        default="auto",
        help="First output index. Use 'auto' to continue after the highest existing image index for this session. Default: auto.",
    )
    parser.add_argument(
        "--storage-id",
        default="mcap",
        help="rosbag2 storage id. Default: mcap.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output PNG files.",
    )
    return parser.parse_args()

#Get picture time stamp in seconds
def image_time_seconds(msg: Image, fallback_timestamp_ns: int) -> float:
    stamp = msg.header.stamp
    if stamp.sec != 0 or stamp.nanosec != 0:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9
    return float(fallback_timestamp_ns) * 1e-9

#Transform raw ROS bytes to NumPy array
def reshape_image_data(msg: Image, dtype, channels: int):
    data = np.frombuffer(msg.data, dtype=dtype)

    if channels == 1:
        row_items = msg.step // dtype.itemsize
        image = data.reshape((msg.height, row_items))[:, : msg.width]
    else:
        row_items = msg.step // dtype.itemsize
        row_pixels = row_items // channels
        image = data.reshape((msg.height, row_pixels, channels))[:, : msg.width, :]

    return image.copy()

#Adjust image encoding to be BGR (used by cv2)
def image_msg_to_cv_image(msg: Image):
    encoding = msg.encoding

    if encoding not in SUPPORTED_ENCODINGS:
        raise ValueError(
            f"Unsupported encoding '{encoding}'. Supported encodings: "
            f"{', '.join(sorted(SUPPORTED_ENCODINGS))}"
        )

    if encoding == "rgb8":
        image = reshape_image_data(msg, np.dtype(np.uint8), 3)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if encoding == "bgr8":
        return reshape_image_data(msg, np.dtype(np.uint8), 3)

    if encoding == "rgba8":
        image = reshape_image_data(msg, np.dtype(np.uint8), 4)
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)

    if encoding == "bgra8":
        return reshape_image_data(msg, np.dtype(np.uint8), 4)

    if encoding == "mono8":
        return reshape_image_data(msg, np.dtype(np.uint8), 1)

#Write png image to storage
def write_png(path: Path, image: np.ndarray):
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")

#Open ROS2 bag for reading
def open_bag_reader(bag_path: Path, storage_id: str):
    storage_options = rosbag2_py.StorageOptions(
        uri = str(bag_path),
        storage_id=storage_id,
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr", 
        output_serialization_format="cdr",
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    return reader

def find_next_index(output_dir: Path, session_name: str):
    max_index = -1
    prefix = f"{session_name}_"

    for path in output_dir.glob(f"{prefix}*.png"):
        index_text = path.stem.removeprefix(prefix)

        if not index_text.isdigit():
            continue

        max_index = max(max_index, int(index_text))

    return max_index + 1

def main():
    args = parse_args()

    if args.every_nth < 1:
        raise ValueError("--every-nth must be >= 1")
    if args.min_interval < 0.0:
        raise ValueError("--min-interval must be >= 0.0")
    if args.max_images < 0:
        raise ValueError("--max-images must be >= 0")

    bag_path = Path(args.bag)
    output_dir = Path(args.output_dir)
    session_name = args.session_name or bag_path.name

    if not bag_path.exists():
        raise FileNotFoundError(f"Bag directory does not exist: {bag_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.start_index == "auto":
        start_index = find_next_index(output_dir, session_name)
    else:
        start_index = int(args.start_index)
        if start_index < 0:
            raise ValueError("--start-index must be >= 0 or 'auto'")

    reader = open_bag_reader(bag_path, args.storage_id)
    topics = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}

    if args.topic not in topics:
        raise RuntimeError(
            f"Topic '{args.topic}' was not found in {bag_path}.\n"
        )

    if topics[args.topic] != "sensor_msgs/msg/Image":
        raise RuntimeError(
            f"Topic '{args.topic}' has type '{topics[args.topic]}', "
            "but this script expects sensor_msgs/msg/Image."
        )
    
    seen = 0
    saved = 0
    skipped_interval = 0
    last_saved_time = None

    while reader.has_next():
        topic, serialized_data, timestamp_ns = reader.read_next()
        if topic != args.topic:
            continue

        seen += 1
        if (seen - 1) % args.every_nth != 0:
            continue

        msg = deserialize_message(serialized_data, Image)
        msg_time = image_time_seconds(msg, timestamp_ns)

        if (last_saved_time is not None
            and args.min_interval > 0.0
            and msg_time - last_saved_time < args.min_interval
        ):
            skipped_interval += 1
            continue

        image = image_msg_to_cv_image(msg)
        output_index = start_index + saved
        output_path = output_dir / f"{session_name}_{output_index:06d}.png"

        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output file already exists: {output_path}. "
                "Use --overwrite or choose a different --output/--prefix."
            )

        write_png(output_path, image)
        saved += 1
        last_saved_time = msg_time

        if args.max_images and saved >= args.max_images:
            break

    print(f"Bag: {bag_path}")
    print(f"Topic: {args.topic}")
    print(f"Messages seen on topic: {seen}")
    print(f"Images saved: {saved}")

    if skipped_interval:
        print(f"Skipped by --min-interval: {skipped_interval}")
    print(f"Output directory: {output_dir}")
    print(f"Session: {session_name}")
    print(f"Start index: {start_index}")

    if saved == 0:
        raise RuntimeError("No images were saved.")

if __name__ == "__main__":
    main()
