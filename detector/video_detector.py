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
    A class for detecting objects in videos using a YOLO model.

    This class handles loading the model, processing video frames, drawing detection results,
    and saving the processed video. It allows specifying the target class ('person')
    and configuring inference settings such as confidence and device.

    Attributes:
        model (YOLO): Loaded YOLO model for object detection.
        model_path (str | Path | YOLO): Path for YOLO model or YOLO object for lazy load.
        class_id (int): ID of the object class to detect (default is 0 for 'person').
        cap (cv2.VideoCapture): OpenCV video capture object.
        out (cv2.VideoWriter): OpenCV video writer object.
        device (str): Inference device ('cpu' or 'cuda').
        conf (float): Confidence threshold for detection.
        frame_count (int): Total number of frames in the input video.
        fps (int): Frame rate of the input video.
        width (int): Width of the video frame.
        height (int): Height of the video frame.
        filename (str): Name of the source video file (without extension).
        output_path (Path): Full path to the saved output video file.

    Methods:
        __init__: Initializes the detector with a given model and class ID.
        load_model: Load and set the model and class ID for detection.
        load_detect: Combines model setup and inference into a single call.
        inference: Runs inference on a video and saves the result with bounding boxes.
        _open_video: Opens the video file and get its properties.
        _init_writer: Prepares the video writer for output.
        _process_frame: Runs inference and annotation for a single frame.
        _release: Releases video resources after processing.
        parse_args: Static method to parse command-line arguments.

    Examples:
        >>> detector = VideoCrowdDetector("path/to/weight", class_id=0)
        >>> detector.inference("path/to/video", save="path/to/save")
    """
    def __init__(self, model: Union[str, Path, YOLO] = 'yolo11n.pt',
                 class_id: int = 0
                 ) -> None:
        """
        Initialize the VideoCrowdDetector with a YOLO model and class ID.

        Args:
            model (Union[str, Path, YOLO]): Path to the YOLO model or a loaded YOLO object.
            class_id (int): The class ID to detect (0 for 'person').
        """
        self.model = model if isinstance(model, YOLO) else None
        self.model_path = model
        self.class_id = class_id

        self.cap = None
        self.out = None
        self.device = 'cpu'
        self.conf = 0.5
        self.frame_count = 0
        self.fps = 0
        self.width = 0
        self.height = 0

        self.filename = ''
        self.output_path = None

    def load_model(self, model: Optional[Union[str, Path, YOLO]] = None,
                   class_id: int = 0
                   ) -> None:
        """
        Set or load the YOLO model and class ID for detection.

        Args:
            model (Optional[Union[str, Path, YOLO]]): Path to the model weights or a YOLO object.
            class_id (int): The class ID to detect.
        """
        if model is None:
            model = self.model_path
            class_id = self.class_id

        if not isinstance(model, YOLO):
            model = YOLO(model)

        self.model = model
        self.class_id = class_id

    def load_detect(self,
                    source: Union[str, Path],
                    model: Union[str, Path, YOLO] = 'yolo11n.pt',
                    class_id: int = 0,
                    save: Union[str, Path] = './output',
                    device: str = 'cpu',
                    conf: float = 0.5
                    ) -> None:
        """
        Load the model and run detection on the given video source.

        This is a convenience method that calls set_model() and then inference().

        Args:
            source (Union[str, Path]): Path to the input video file.
            model (Union[str, Path, YOLO]): Path to the model weights or a YOLO object.
            class_id (int): The class ID to detect.
            save (Union[str, Path]): Directory to save the output video.
            device (str): Device to use for inference.
            conf (float): Confidence threshold for detections.
        """
        self.load_model(model, class_id)
        self.inference(source, save, device, conf)

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
        if not self.model:
            self.load_model(self.model_path, self.class_id)
        self.device = device
        self.conf = conf

        self._open_video(source)
        self._init_writer(save)

        current_frame = 1
        with tqdm(total=self.frame_count, desc="Video processing", unit="frame") as pbar:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    LOGGER.debug('End of video or error occurred')
                    break

                labeled_frame = self._process_frame(frame)
                self.out.write(labeled_frame)

                LOGGER.debug(f'frame {current_frame}/{self.frame_count} processed')
                current_frame += 1
                pbar.update(1)

        self._release()

    def _open_video(self, source: Union[str, Path]) -> None:
        """
        Open the video file and initialize capture properties.

        Args:
            source (Union[str, Path]): Path to input video file.
        """
        source_path = Path(source)
        self.filename = source_path.stem
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            LOGGER.error(f"Failed to open video file: {source_path}")
            raise FileNotFoundError(f"Failed to open video file: {source_path}")

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def _init_writer(self, save_dir: Union[str, Path]) -> None:
        """
        Initialize the video writer for saving output.

        Args:
            save_dir (Union[str, Path]): Directory where output video will be stored.
        """
        suffix, fourcc = (".mp4", "mp4v") if MAC else (".avi", "XVID") if WINDOWS else (".avi", "MJPG")
        fourcc = cv2.VideoWriter_fourcc(*fourcc)

        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        self.output_path = Path(save_path, self.filename).with_suffix(suffix)

        self.out = cv2.VideoWriter(
            filename=str(self.output_path),
            fourcc=fourcc,
            fps=self.fps,
            frameSize=(self.width, self.height)
            )

        if not self.out.isOpened():
            LOGGER.error(f"Failed to create video file: {self.output_path}")
            raise IOError(f"Failed to create video file: {self.output_path}")

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Run inference and draw detections on a single frame.

        Args:
            frame (np.ndarray): Single frame in BGR format from OpenCV.

        Returns:
            np.ndarray: Annotated frame with bounding boxes and labels.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pred = self.model(rgb_frame, conf=self.conf, device=self.device, verbose=False)[0]

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

        return plotter.result()

    def _release(self) -> None:
        """
        Release video writer and capture objects after processing is complete.
        """
        self.out.release()
        self.cap.release()
        LOGGER.info(f'Video saved successfully at {self.output_path}')

    @staticmethod
    def parse_args() -> argparse.Namespace:
        """
        Parse command-line arguments for the detector.

        Returns:
            argparse.Namespace: Parsed arguments including weights, source, class_id, etc.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str, default="yolo11n.pt", help="path to model weights")
        parser.add_argument("--class_id", type=int, default=0, help="ID of the detected class")
        parser.add_argument("--source", type=str, required=True, help="path to source video")
        parser.add_argument("--save", type=str, default="./output", help="path to save result video")
        parser.add_argument("--device", type=str, default="cpu", help="cuda device, 0 or 0,1,2,3 or cpu")
        parser.add_argument("--conf", type=float, default=0.5, help="confidence score for detect")
        return parser.parse_args()
