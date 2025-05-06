from Frame import Frame
from Datatypes import *
from custom_types import FrameID, FeatureID
import cv2
import numpy as np


class FrameGenerator:
    def __init__(self, detector):
        self.next_image_counter = 0
        self.detector = detector

    def make_frame(self, image) -> Frame:
        """
        Create a frame by extracting features from the provided image.

        This method should only be called once for each image.
        Each of the extracted features will be assigned a unique
        id, whic will help with tracking of individual features
        later in the pipeline.
        """
        # Create a frame and assign it a unique id.
        frame = Frame(image)
        frame.id = FrameID(self.next_image_counter)
        self.next_image_counter += 1

        # Extract features
        frame.keypoints, frame.descriptors = self.detector.detectAndCompute(
            frame.image, None
        )
        enumerated_features = enumerate(zip(frame.keypoints, frame.descriptors))

        # Visualize Features
        if False:
            print("waiting")
            tmpIMG = np.copy(image)

            tmpIMG = cv2.drawKeypoints(
                image,
                frame.keypoints,
                tmpIMG,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )
            cv2.imwrite("output/Features.png", tmpIMG)
            # cv2.imshow("Features", tmpIMG)
            # cv2.waitKey(100000)

        # Save features in a list with the following elements
        # keypoint, descriptor, feature_id
        # where the feature_id refers to the image id and the feature
        # number.
        frame.features = [
            Feature(
                keypoint,
                descriptor,
                FeatureID(
                    (frame.id, idx),
                ),
            )
            for (idx, (keypoint, descriptor)) in enumerated_features
        ]

        return frame
