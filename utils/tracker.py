import numpy as np
from scipy.spatial.distance import cdist

class PlayerTracker:
    def __init__(self, max_distance=50):
        self.next_id = 0
        self.tracks = []
        self.max_distance = max_distance

    def update(self, detections):
        updated_tracks = []
        if len(self.tracks) == 0:
            for det in detections:
                updated_tracks.append({'id': self.next_id, 'bbox': det})
                self.next_id += 1
        else:
            prev_bboxes = np.array([t['bbox'] for t in self.tracks])
            curr_bboxes = np.array(detections)

            if len(curr_bboxes) == 0:
                self.tracks = []
                return []

            dists = cdist(prev_bboxes[:, :2], curr_bboxes[:, :2])
            matched_indices = np.argmin(dists, axis=1)

            used = set()
            for i, t in enumerate(self.tracks):
                j = matched_indices[i]
                if dists[i][j] < self.max_distance and j not in used:
                    updated_tracks.append({'id': t['id'], 'bbox': curr_bboxes[j]})
                    used.add(j)

            for j, det in enumerate(curr_bboxes):
                if j not in used:
                    updated_tracks.append({'id': self.next_id, 'bbox': det})
                    self.next_id += 1

        self.tracks = updated_tracks
        return updated_tracks
