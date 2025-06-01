from detector.video_detector import VideoCrowdDetector

if __name__ == '__main__':
    detector = VideoCrowdDetector()
    detector.load_detect(**vars(VideoCrowdDetector.parse_args()))
