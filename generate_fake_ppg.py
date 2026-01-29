import cv2
import numpy as np
import os
from pathlib import Path

def generate_ppg_from_video(video_path, output_name='ppg_label.npy', fps=30, duration_seconds=None):
    """
    Generate synthetic PPG signal from a video file.
    
    Args:
        video_path: Path to the video file
        output_name: Name of the output .npy file (default: 'ppg_label.npy')
        fps: Expected frames per second of the video
        duration_seconds: Optional - limit to specific duration in seconds
    
    Returns:
        ppg_signal: Generated PPG signal array
    """
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found!")
        return None
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return None
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {video_fps}")
    print(f"  Resolution: {frame_width}x{frame_height}")
    
    # Calculate how many frames to process
    if duration_seconds:
        frames_to_process = min(total_frames, int(duration_seconds * video_fps))
    else:
        frames_to_process = total_frames
    
    ppg_signal = []
    frame_count = 0
    
    print(f"\nProcessing {frames_to_process} frames...")
    
    while cap.isOpened() and frame_count < frames_to_process:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Convert frame to HSV to extract skin color information
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Extract mean green channel (PPG is typically measured from green channel)
        # This simulates extracting heart rate signal from the green color variations
        green_channel = frame[:, :, 1]  # Green channel in BGR
        
        # Calculate mean intensity of the frame
        mean_intensity = np.mean(green_channel)
        
        ppg_signal.append(mean_intensity)
        frame_count += 1
        
        if (frame_count + 1) % 30 == 0:
            print(f"  Processed {frame_count + 1}/{frames_to_process} frames")
    
    cap.release()
    
    # Convert to numpy array
    ppg_signal = np.array(ppg_signal, dtype=np.float32)
    
    # Normalize the signal
    ppg_signal = (ppg_signal - np.mean(ppg_signal)) / (np.std(ppg_signal) + 1e-8)
    
    # Add some realistic PPG oscillations (simulating heart rate)
    t = np.arange(len(ppg_signal))
    heart_rate_signal = 0.5 * np.sin(2 * np.pi * t / (video_fps * 0.5))  # ~60 BPM oscillation
    ppg_signal = ppg_signal + heart_rate_signal
    
    # Save to file
    output_path = output_name
    np.save(output_path, ppg_signal)
    
    print(f"\nPPG signal generated successfully!")
    print(f"  Length: {len(ppg_signal)} samples")
    print(f"  Duration: {len(ppg_signal) / video_fps:.2f} seconds")
    print(f"  Saved to: {output_path}")
    
    return ppg_signal


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python generate_fake_ppg.py <video_path> [output_name] [duration_seconds]")
        print("\nExample:")
        print("  python generate_fake_ppg.py video.mp4")
        print("  python generate_fake_ppg.py video.mp4 custom_ppg.npy")
        print("  python generate_fake_ppg.py video.mp4 custom_ppg.npy 10")
        
        # Demo mode - create a simple synthetic PPG if no video provided
        print("\n--- Demo Mode (no video provided) ---")
        demo_ppg = np.random.randn(300) * 0.1
        t = np.arange(300)
        demo_ppg += np.sin(2 * np.pi * t / 30)  # ~60 BPM oscillation at 30 fps
        np.save('ppg_label.npy', demo_ppg)
        print("Created demo ppg_label.npy with 300 synthetic samples")
        return
    
    video_path = sys.argv[1]
    output_name = sys.argv[2] if len(sys.argv) > 2 else 'ppg_label.npy'
    duration_seconds = float(sys.argv[3]) if len(sys.argv) > 3 else None
    
    generate_ppg_from_video(video_path, output_name, duration_seconds=duration_seconds)


if __name__ == '__main__':
    main()
