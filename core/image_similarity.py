import os
import numpy as np
import concurrent.futures
from skimage.util import compare_images
from skimage.transform import resize
from skimage.color import rgb2gray


class ImageProcessor:
    def __init__(self):
        self.p = "hi"


    # Function to compute similarity using difference comparison
 
    def compute_similarity_diff(self,img1, img2):
        # Resize img2 to the same shape as img1, if necessary
        target_shape = img1.shape
        img2_resized = resize(img2, target_shape, anti_aliasing=True)
        
        # Difference comparison
        diff_image = compare_images(img1, img2_resized, method='diff')
        similarity_score = 1 - np.mean(diff_image)  # Closer to 1 means more similar
        return similarity_score

    # Function to compare two images and compute the similarity score
    def compare_images_between_pdfs(self,img1, img2):
        similarity_score = self.compute_similarity_diff(img1, img2)
        return similarity_score


    # Function to run multithreaded comparisons between all PDFs and calculate the top 10 similarity scores average
    def threaded_pdf_comparison(self,pdf_images_1, pdf_images_2, max_workers=4):
        similarity_scores = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # List of tasks for each pair of images
            tasks = [
                executor.submit(self.compare_images_between_pdfs, img1, img2)
                for img1 in pdf_images_1 for img2 in pdf_images_2
            ]
            
            # Retrieve the results as they are completed
            for future in concurrent.futures.as_completed(tasks):
                similarity_scores.append(future.result())
        
        # Sort the scores in descending order and select the top 10
        top_10_scores = sorted(similarity_scores, reverse=True)[:10]
        
        # Calculate the average of the top 10 scores
        if top_10_scores:
            avg_top_10_similarity = np.mean(top_10_scores)
        else:
            avg_top_10_similarity = 0  # Default to 0 if there are no similarity scores

        return avg_top_10_similarity
