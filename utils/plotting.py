from typing import Optional, Tuple

import cv2
import numpy as np


class Plotter:
    """
    A utility class for drawing bounding boxes and labels on images.

    This class is typically used in conjunction with object detection models YOLO
    to visualize detection results by plotting bounding boxes and class labels onto frames.

    Attributes:
        img (np.ndarray): The input image to plot.
        line_width (int): width of the bounding box lines.
        alpha (float): Transparency level for the label background.
        font_thickness (int): Thickness of the font text.
        font_scale (float): Scale factor of the font text.

    Methods:
        __init__: Initialize the Plotter object with image, line width and font size.
        plot_box: Plot bounding box to image.
        result: Returns plotted image as numpy array.

    Examples:
        >>> plotter = Plotter(img)
        >>> for box, label in zip(boxes, labels):
        ...     plotter.plot_box(box, label)
        >>> plotter_image = plotter.result()
    """
    def __init__(self,
                 img: np.ndarray,
                 line_width: Optional[int] = None,
                 alpha: float = 0.4
                 ) -> None:
        """
        Initialize the Plotter object.

        Args:
            img (np.ndarray): Image array in BGR format.
            line_width (int, optional): Line thickness for plotting boxes.
            alpha (float): Transparency level for the label background.
        """
        self.img = img
        self.line_width = line_width or max(round(sum(img.shape[:2]) / 1200), 1)
        self.alpha = alpha
        self.font_thickness = max(self.line_width - 4, 1)
        self.font_scale = max(round(0.0005 * img.shape[0], 1), 0.7)

    def plot_box(self, bbox,
                 label: str = '',
                 color: Tuple[int, int, int] = (0, 0, 255),
                 text_color: Tuple[int, int, int] = (255, 255, 255),
                 ) -> None:
        """
        Plot a bounding box and optional label on the image.

        Args:
            bbox (np.ndarray): Bounding box coordinates (x1, y1, x2, y2).
            label (str): Label to display near the box.
            color (Tuple[int, int, int]): Box color in BGR.
            text_color (Tuple[int, int, int]): Text color in BGR.
        """
        bbox = bbox.astype(int)
        p1, p2 = (bbox[0], bbox[1]), (bbox[2], bbox[3])

        cv2.rectangle(self.img,
                      p1,
                      p2,
                      color=color,
                      thickness=self.line_width,
                      lineType=cv2.LINE_AA)

        if label:
            (w, h), _ = cv2.getTextSize(label,
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=self.font_scale,
                                        thickness=self.font_thickness)
            h += 3
            is_outside = p1[1] - h >= 0
            if p1[0] > self.img.shape[1] - w:
                p1 = self.img.shape[1] - w, p1[1]
            text_start_coord = (p1[0], p1[1] - 2) if is_outside else (p1[0], p1[1] + h + 2)

            rect_p1 = (p1[0], p1[1] - h if is_outside else p1[1])
            rect_p2 = (p1[0] + w, p1[1]) if is_outside else (p1[0] + w, p1[1] + h + 3)

            overlay = self.img.copy()
            cv2.rectangle(overlay, rect_p1, rect_p2, color, thickness=-1)
            cv2.addWeighted(overlay, self.alpha, self.img, 1 - self.alpha, 0, self.img)

            cv2.putText(
                self.img,
                label,
                text_start_coord,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=self.font_scale,
                color=text_color,
                thickness=self.font_thickness,
                lineType=cv2.LINE_AA
            )

    def result(self) -> np.ndarray:
        """
        Return the image with all drawn plots.

        Returns:
            np.ndarray: Plotted image.
        """
        return self.img
