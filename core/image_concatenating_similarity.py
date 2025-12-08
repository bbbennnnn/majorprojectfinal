import cv2
import numpy as np

class ImageProcessor_Concatenated:
    def __init__(self, saturation_threshold=5, min_width=50, min_height=50):
        self.saturation_threshold = saturation_threshold
        self.min_width = min_width
        self.min_height = min_height

    def calculate_color_saturation(self, image):
        """Calculate the average saturation of an image."""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert image to HSV
        saturation = hsv_image[:, :, 1]  # Extract the saturation channel
        return np.mean(saturation)  # Return the average saturation

    def is_significant_image(self, image):
        """Check if an image meets minimum dimensions."""
        height, width, _ = image.shape
        return width >= self.min_width and height >= self.min_height

    def process_images(self, raw_images):
        """Filter significant images and calculate saturation."""
        image_info = []
        for index, image in raw_images:
            if not self.is_significant_image(image):
                continue
            saturation = self.calculate_color_saturation(image)
            image_info.append((saturation, index, image))
        return sorted(image_info, key=lambda x: x[0], reverse=False)

    def find_close_scores(self, list1, list2):
        """Find image indices with close color saturation scores."""
        pointer1, pointer2 = 0, 0
        list1_indexes, list2_indexes = [], []

        while pointer1 < len(list1) and pointer2 < len(list2):
            score1, index1, _ = list1[pointer1]
            score2, index2, _ = list2[pointer2]

            diff = abs(score1 - score2)

            if diff <= self.saturation_threshold:
                list1_indexes.append(index1)
                list2_indexes.append(index2)
                pointer1 += 1
                pointer2 += 1
            elif score1 > score2:
                pointer1 += 1
            else:
                pointer2 += 1

        return list1_indexes, list2_indexes

    def concatenate_images(self, images):
        """Concatenate a list of images vertically."""
        if len(images) == 0:
            return None

        min_width = min(image.shape[1] for image in images)
        resized_images = [
            cv2.resize(image, (min_width, int(image.shape[0] * (min_width / image.shape[1]))))
            for image in images
        ]
        return np.vstack(resized_images)

    def compare_pdfs_concatenated(self, pdf1_images, pdf2_images):
        """Compare two PDFs by concatenating and analyzing images."""
        # Process images to filter and calculate saturation
        pdf1_processed = self.process_images(pdf1_images)
        pdf2_processed = self.process_images(pdf2_images)

        # Find close matches based on saturation
        close_indexes1, close_indexes2 = self.find_close_scores(pdf1_processed, pdf2_processed)

        # Filter images based on close indices
        filtered_images1 = [img for _, idx, img in pdf1_processed if idx in close_indexes1]
        filtered_images2 = [img for _, idx, img in pdf2_processed if idx in close_indexes2]

        # Concatenate filtered images
        concatenated_image1 = self.concatenate_images(filtered_images1)
        concatenated_image2 = self.concatenate_images(filtered_images2)

        if concatenated_image1 is None or concatenated_image2 is None:
            print("Concatenation failed for one or both PDFs.")
            return 0  # No similarity if images cannot be concatenated

        # Compare concatenated images using SIFT
        return self.compare_images_sift(concatenated_image1, concatenated_image2)

    def compare_images_sift(self, image1, image2):
        """Compare two concatenated images using SIFT."""
        # Convert to grayscale after concatenation
        image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(image1_gray, None)
        keypoints2, descriptors2 = sift.detectAndCompute(image2_gray, None)

        # Handle cases with no descriptors
        if descriptors1 is None or descriptors2 is None:
            print("No descriptors found for one or both images.")
            return 0  # Return 0 similarity if no descriptors are found

        # Match descriptors using Brute-Force Matcher
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)

        # Compute similarity as the ratio of matches to the total number of keypoints
        similarity_score = len(matches) / max(len(keypoints1), len(keypoints2))
        return similarity_score
