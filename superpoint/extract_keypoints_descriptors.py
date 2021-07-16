from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf  # noqa: E402

from settings import EXPER_PATH  # noqa: E402


def extract_superpoint_keypoints_and_descriptors(keypoint_map, descriptor_map, keep_k_points):
    def select_k_best(points, k):
        """Select the k most probable points (and strip their proba).
        points has shape (num_points, 3) where the last coordinate is the proba."""
        sorted_prob = points[points[:, 2].argsort(), :2]
        start = min(k, points.shape[0])
        return sorted_prob[-start:, :]

    # Extract keypoints
    keypoints = np.where(keypoint_map > 0)
    prob = keypoint_map[keypoints[0], keypoints[1]]
    keypoints = np.stack([keypoints[0], keypoints[1], prob], axis=-1)

    keypoints = select_k_best(keypoints, keep_k_points)
    keypoints = keypoints.astype(int)

    # Get descriptors for keypoints
    desc = descriptor_map[keypoints[:, 0], keypoints[:, 1]]

    # Convert from just pts to cv2.KeyPoints
    keypoints = [cv2.KeyPoint(float(p[1]), float(p[0]), 1) for p in keypoints]

    return keypoints, desc


def preprocess_image(img_file, img_size, resize_image):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if resize_image:
        img = cv2.resize(img, img_size)
    img_orig = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, 2)
    img = img.astype(np.float32)
    img_preprocessed = img / 255.0

    return img_preprocessed, img_orig


def get_superpoint_kp_desc(
    img_file,
    weights_name: str = "sp_v6",
    resize_image: bool = True,
    height: int = 480,
    width: int = 640,
    keep_kp_best: int = 1000,
):
    """
    superpoint_generator [summary]

    Args:
        img_file ([type]): [description]
        weights_name (str, optional): [weights_name]. Defaults to "sp_v6".
        resize_image (bool, optional): [whether to resize image]. Defaults to True.
        height (int, optional): [The height in pixels to resize the images to]. Defaults to 480.
        width (int, optional): [The width in pixels to resize the images to]. Defaults to 640.
        keep_kp_best (int, optional): [Maximum number of keypoints to keep]. Defaults to 1000.
    """
    img_size = (width, height)

    weights_root_dir = Path(EXPER_PATH, "saved_models")
    weights_root_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path(weights_root_dir, weights_name)

    graph = tf.Graph()
    with tf.compat.v1.Session(graph=graph) as sess:
        tf.compat.v1.saved_model.load(sess, [tf.saved_model.SERVING], str(weights_dir))

        input_img_tensor = graph.get_tensor_by_name("superpoint/image:0")
        output_prob_nms_tensor = graph.get_tensor_by_name("superpoint/prob_nms:0")
        output_desc_tensors = graph.get_tensor_by_name("superpoint/descriptors:0")

        img, img_orig = preprocess_image(img_file, img_size, resize_image)
        out = sess.run(
            [output_prob_nms_tensor, output_desc_tensors], feed_dict={input_img_tensor: np.expand_dims(img, 0)}
        )
        keypoint_map = np.squeeze(out[0])
        descriptor_map = np.squeeze(out[1])
        kp, desc = extract_superpoint_keypoints_and_descriptors(keypoint_map, descriptor_map, keep_kp_best)
    return kp, desc

if __name__ == "__main__":
    kp_super, desc = get_superpoint_kp_desc(img_file="/content/drive/MyDrive/ColabNotebooks/superpoint/data/val2014/COCO_val2014_000000000164.jpg", resize_image=False)
    img = cv2.imread('/content/drive/MyDrive/ColabNotebooks/superpoint/data/val2014/COCO_val2014_000000000164.jpg')
    sift = cv2.SIFT_create()
    kp = sift.detect(img,None)
    img=cv2.drawKeypoints(img,kp,img)
    cv2.imwrite('sift_keypoints.jpg',img)
    img = cv2.imread('/content/drive/MyDrive/ColabNotebooks/superpoint/data/val2014/COCO_val2014_000000000164.jpg')
    img=cv2.drawKeypoints(img,kp_super,img)
    cv2.imwrite('superpoint_keypoints.jpg',img)