"""
Gesture detector for BISINDO Bridge.
Detects fist and open palm gestures from MediaPipe hand landmarks using pure geometry.
"""

import numpy as np
from typing import List, Optional, Union


def _calculate_distances(landmarks: np.ndarray) -> np.ndarray:
    """
    Calculate Euclidean distances from each landmark to the wrist (landmark 0).

    Args:
        landmarks: Array of shape (21, 3) with x, y, z coordinates

    Returns:
        Array of distances from each landmark to wrist
    """
    wrist = landmarks[0]
    distances = np.linalg.norm(landmarks - wrist, axis=1)
    return distances


def _normalize_by_hand_size(landmarks: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """
    Normalize distances by hand size (max distance between any two landmarks).

    Args:
        landmarks: Array of shape (21, 3)
        distances: Raw distances to wrist

    Returns:
        Normalized distances
    """
    # Calculate all pairwise distances
    pairwise = np.linalg.norm(landmarks[:, np.newaxis] - landmarks[np.newaxis, :], axis=2)
    max_distance = np.max(pairwise)

    if max_distance < 1e-6:  # Avoid division by zero
        return distances

    return distances / max_distance


def is_fist(landmarks_21x3: Union[np.ndarray, List]) -> bool:
    """
    Detect if hand is making a fist.

    A fist is detected when all fingertips (indices 4, 8, 12, 16, 20) are close
    to the wrist (index 0).

    Args:
        landmarks_21x3: 21 landmarks × 3 coordinates (x, y, z)

    Returns:
        True if fist detected, False otherwise
    """
    if landmarks_21x3 is None or len(landmarks_21x3) == 0:
        return False

    landmarks = np.array(landmarks_21x3)
    if landmarks.shape != (21, 3):
        return False

    distances = _calculate_distances(landmarks)
    normalized = _normalize_by_hand_size(landmarks, distances)

    # Fingertip indices
    fingertips = [4, 8, 12, 16, 20]
    fingertip_distances = normalized[fingertips]

    # Fist: all fingertips close to wrist (threshold < 0.3)
    mean_distance = np.mean(fingertip_distances)
    return mean_distance < 0.3


def is_open_palm(landmarks_21x3: Union[np.ndarray, List]) -> bool:
    """
    Detect if hand is showing an open palm.

    An open palm is detected when all fingertips are far from the wrist AND
    spread apart from each other.

    Args:
        landmarks_21x3: 21 landmarks × 3 coordinates (x, y, z)

    Returns:
        True if open palm detected, False otherwise
    """
    if landmarks_21x3 is None or len(landmarks_21x3) == 0:
        return False

    landmarks = np.array(landmarks_21x3)
    if landmarks.shape != (21, 3):
        return False

    distances = _calculate_distances(landmarks)
    normalized = _normalize_by_hand_size(landmarks, distances)

    # Fingertip indices
    fingertips = [4, 8, 12, 16, 20]
    fingertip_distances = normalized[fingertips]

    # Check 1: All fingertips far from wrist (threshold > 0.5)
    if not np.all(fingertip_distances > 0.5):
        return False

    # Check 2: Fingertips spread apart (inter-fingertip distances > 0.1)
    fingertip_landmarks = landmarks[fingertips]
    pairwise = np.linalg.norm(
        fingertip_landmarks[:, np.newaxis] - fingertip_landmarks[np.newaxis, :],
        axis=2
    )

    # Get upper triangle (excluding diagonal)
    upper_tri = pairwise[np.triu_indices(len(fingertips), k=1)]
    mean_spread = np.mean(upper_tri)

    return mean_spread > 0.1


def get_hand_state(landmarks_21x3: Optional[Union[np.ndarray, List]]) -> str:
    """
    Determine the current hand state based on landmarks.

    Args:
        landmarks_21x3: 21 landmarks × 3 coordinates, or None

    Returns:
        One of: "fist", "palm", "signing", "none"
    """
    if landmarks_21x3 is None or len(landmarks_21x3) == 0:
        return "none"

    # Check fist first (highest priority)
    if is_fist(landmarks_21x3):
        return "fist"

    # Check open palm
    if is_open_palm(landmarks_21x3):
        return "palm"

    # If hand is detected but not fist or palm, it's signing
    return "signing"
