"""
boundary_setup.py - Interactive Geofence Boundary Drawing Tool

Usage:
    python boundary_setup.py --camera_id cam1 --source 0

Instructions:
    1. Click on the video to mark boundary points
    2. Press 'c' to clear all points
    3. Press 's' to save the boundary
    4. Press 'q' to quit without saving
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


class BoundaryDrawer:
    """Interactive polygon drawing tool for geofence boundaries."""

    def __init__(self, camera_id: str, source: int | str):
        self.camera_id = camera_id
        self.source = source
        self.points: List[Tuple[int, int]] = []
        self.drawing = False
        self.window_name = f"Boundary Setup - Camera: {camera_id}"

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to add boundary points."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            print(f"Point added: ({x}, {y}) | Total points: {len(self.points)}")

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw the boundary polygon and points on the frame."""
        overlay = frame.copy()

        # Draw lines between points
        if len(self.points) > 1:
            for i in range(len(self.points) - 1):
                cv2.line(overlay, self.points[i], self.points[i + 1], (0, 255, 0), 2)
            # Close the polygon
            if len(self.points) > 2:
                cv2.line(overlay, self.points[-1], self.points[0], (0, 255, 0), 2)

        # Draw filled polygon with transparency
        if len(self.points) > 2:
            pts = np.array(self.points, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Draw points as circles
        for i, point in enumerate(self.points):
            cv2.circle(frame, point, 5, (0, 0, 255), -1)
            cv2.putText(
                frame,
                str(i + 1),
                (point[0] + 10, point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        return frame

    def save_boundary(self) -> bool:
        """Save the boundary points to a JSON file."""
        if len(self.points) < 3:
            print("❌ Need at least 3 points to define a boundary!")
            return False

        # Create boundaries directory if it doesn't exist
        boundaries_dir = Path("boundaries")
        boundaries_dir.mkdir(exist_ok=True)

        # Save to JSON
        filepath = boundaries_dir / f"{self.camera_id}_boundary.json"
        data = {
            "camera_id": self.camera_id,
            "source": str(self.source),
            "points": self.points,
            "num_points": len(self.points),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"✅ Boundary saved to: {filepath}")
        return True

    def run(self):
        """Main loop for boundary drawing."""
        # Open video source
        if isinstance(self.source, str) and not self.source.isdigit():
            cap = cv2.VideoCapture(self.source)
        else:
            cap = cv2.VideoCapture(int(self.source))

        if not cap.isOpened():
            print(f"❌ Cannot open camera/video source: {self.source}")
            return

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\n" + "=" * 60)
        print("🎯 BOUNDARY SETUP MODE")
        print("=" * 60)
        print("📌 Click on the video to mark boundary points")
        print("⌨️  Press 'c' to clear all points")
        print("💾 Press 's' to save the boundary")
        print("🚪 Press 'q' to quit without saving")
        print("=" * 60 + "\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  End of video or cannot read frame")
                break

            # Flip frame horizontally for webcams (fixes mirror image)
            if isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit()):
                frame = cv2.flip(frame, 1)

            # Draw overlay
            display_frame = self.draw_overlay(frame)

            # Add instructions text
            cv2.putText(
                display_frame,
                f"Points: {len(self.points)} | 'c'=clear 's'=save 'q'=quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow(self.window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("🚪 Quit without saving")
                break
            elif key == ord("c"):
                self.points.clear()
                print("🗑️  All points cleared")
            elif key == ord("s"):
                if self.save_boundary():
                    break

        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Draw geofence boundary for a camera")
    parser.add_argument(
        "--camera_id",
        type=str,
        required=True,
        help="Camera identifier (e.g., cam1, entrance_cam)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: camera index (0, 1) or video file path",
    )

    args = parser.parse_args()

    drawer = BoundaryDrawer(camera_id=args.camera_id, source=args.source)
    drawer.run()


if __name__ == "__main__":
    main()