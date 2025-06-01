import argparse
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from utils import LOGGER, MAC, WINDOWS
from utils.plotting import Plotter


class VideoCrowdDetector:
    """
    A class for detecting objects (e.g., people) in videos using a YOLO model.

    This class handles loading the model, processing video frames, drawing detection results,
    and saving the processed video. It allows specifying the target class ('person')
    and configuring inference settings such as confidence and device.

    Attributes:
        model: Loaded YOLO model for object detection.
        class_id: ID of the object class to detect (default is 0 for 'person').

    Methods:
        __init__: Initializes the detector with a given model and class ID.
        set_model: Sets the model and class ID for detection.
        inference: Runs inference on a video and saves the result with bounding boxes.
        load_detect: Shortcut method to set the model and run inference.
        parse_args: Static method to parse command-line arguments.

    Examples:
        >>> detector = VideoCrowdDetector("path/to/weight", class_id=0)
        >>> detector.inference("path/to/video", save="path/to/save")
    """
    def __init__(self, model: Union[str, YOLO, None] = None,
                 class_id: Optional[int] = None) -> None:
        """
        Initialize the VideoCrowdDetector with a YOLO model and class ID.

        Args:
            model (Union[str, YOLO, None]): Path to the YOLO model or a loaded YOLO object.
            class_id (Optional[int]): The class ID to detect (0 for 'person').
        """
        if model:
            self.set_model(model, class_id)

    def set_model(self, model: Union[str, YOLO] = 'yolo11n.pt',
                  class_id: int = 0) -> None:
        """
        Set or replace the YOLO model and class ID for detection.

        Args:
            model (Union[str, YOLO]): Path to the model weights or a YOLO object.
            class_id (int): The class ID to detect.
        """
        self.model = model if type(model) is YOLO else YOLO(model)
        self.class_id = class_id

    def inference(self,
                  source: Union[str, Path],
                  save: Union[str, Path] = './output',
                  device: str = 'cpu',
                  conf: float = 0.5
                  ) -> None:
        """
        Run detection on the given video file and save the output with drawn bboxes.

        Args:
            source (Union[str, Path]): Path to the input video file.
            save (Union[str, Path]): Directory to save the output video.
            device (str): Device to use for inference ('cpu' or 'cuda:0').
            conf (float): Confidence threshold for detections.
        """
        source_path = Path(source)
        filename = source_path.stem

        save_path = Path(save)
        save_path.mkdir(exist_ok=True, parents=True)

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            LOGGER.error(f"Failed to open video file: {source_path}")
            raise FileNotFoundError(f"Failed to open video file: {source_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        suffix, fourcc = (".mp4", "mp4v") if MAC else (".avi", "XVID") if WINDOWS else (".avi", "MJPG")
        fourcc = cv2.VideoWriter_fourcc(*fourcc)
        save_path = Path(save_path, filename).with_suffix(suffix)
        out = cv2.VideoWriter(
            filename=str(save_path),
            fourcc=fourcc,
            fps=fps,
            frameSize=(width, height))

        if not out.isOpened():
            LOGGER.error(f"Couldn't create the video file: {save_path}")
            raise IOError(f"Error create video: {save_path}")

        current_frame = 1
        with tqdm(total=frame_count, desc="Обработка видео", unit="кадр") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    LOGGER.debug('End of video or error occurred')
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pred = self.model(rgb_frame, conf=conf, device=device, verbose=False)[0]

                names = pred.names
                boxes_class_id = pred.boxes.cls.cpu().numpy()
                indices = np.where(boxes_class_id == self.class_id)[0]

                boxes_coords = pred.boxes.xyxy.cpu().numpy()[indices]
                boxes_probs = pred.boxes.conf.cpu().numpy()[indices]
                class_name = names[self.class_id]

                plotter = Plotter(frame)

                for box, prob in zip(boxes_coords, boxes_probs):
                    label = f'{class_name} {prob:.2f}'
                    plotter.plot_box(box, label)

                labeled_frame = plotter.result()
                out.write(labeled_frame)

                LOGGER.debug(f'frame {current_frame}/{frame_count} processed')
                current_frame += 1
                pbar.update(1)

        out.release()
        cap.release()
        LOGGER.info('Video saved successfully')

    def load_detect(self,
                    source: Union[str, Path],
                    model: Union[str, YOLO] = 'yolo11n.pt',
                    class_id: int = 0,
                    save: Union[str, Path] = './output',
                    device: str = 'cpu',
                    conf: float = 0.5):
        """
        Load the model and run detection on the given video source.

        This is a convenience method that calls set_model() and then inference().

        Args:
            source (Union[str, Path]): Path to the input video file.
            model (Union[str, YOLO]): Path to the model weights or a YOLO object.
            class_id (int): The class ID to detect.
            save (Union[str, Path]): Directory to save the output video.
            device (str): Device to use for inference.
            conf (float): Confidence threshold for detections.
        """
        self.set_model(model, class_id)
        self.inference(source, save, device, conf)

    @staticmethod
    def parse_args() -> argparse.Namespace:
        """
        Parse command-line arguments for the detector.

        Returns:
            argparse.Namespace: Parsed arguments including weights, source, class_id, etc.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str, default="yolo11n.pt",
                            help="path to model weights")

        parser.add_argument("--class_id", type=int, default=0,
                            help="ID of the detected class")

        parser.add_argument("--source", type=str, required=True,
                            help="path to source video")

        parser.add_argument("--save", type=str, default="./output",
                            help="path to save result video")

        parser.add_argument("--device", type=str, default="cpu",
                            help="cuda device, 0 or 0,1,2,3 or cpu")

        parser.add_argument("--conf", type=float, default=0.5,
                            help="confidence score for detect")
        return parser.parse_args()
