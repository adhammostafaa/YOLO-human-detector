import argparse
import supervision as sv
from trackers import ByteTrackTracker
from ultralytics import YOLO


def track_video(model_path, video_in, video_out, imgsz=640, conf=0.25, device='cpu', line_orientation='vertical'):
    model = YOLO(model_path, task='detect')
    video_info = sv.VideoInfo.from_video_path(video_in)
    tracker = ByteTrackTracker(
        frame_rate=video_info.fps,
        track_activation_threshold=0.25,
        minimum_consecutive_frames=2
    )

    line_zone = None
    if line_orientation == 'vertical':
        mid_x = video_info.width // 2
        line_zone = sv.LineZone(
            start=sv.Point(mid_x, 0),
            end=sv.Point(mid_x, video_info.height),
            triggering_anchors=[sv.Position.CENTER]
        )
    elif line_orientation == 'horizontal':
        mid_y = video_info.height // 2
        line_zone = sv.LineZone(
            start=sv.Point(0, mid_y),
            end=sv.Point(video_info.width, mid_y),
            triggering_anchors=[sv.Position.CENTER]
        )

    box_ann = sv.BoxAnnotator(thickness=2)
    label_ann = sv.LabelAnnotator(text_scale=0.6, text_thickness=1)
    trace_ann = sv.TraceAnnotator(thickness=2, trace_length=30)
    line_ann = sv.LineZoneAnnotator(thickness=2, text_scale=0.8, text_thickness=2)

    frames = sv.get_video_frames_generator(video_in)
    with sv.VideoSink(video_out, video_info) as sink:
        for frame in frames:
            res = model.predict(frame, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
            dets = sv.Detections.from_ultralytics(res)
            dets = tracker.update(dets)

            if dets.tracker_id is not None:
                dets = dets[dets.tracker_id >= 0]

            if line_zone is not None and len(dets) > 0:
                line_zone.trigger(dets)

            labels = [
                f"#{tid} {c:.2f}" for tid, c in zip(dets.tracker_id, dets.confidence)
            ] if dets.tracker_id is not None else []

            out_frame = trace_ann.annotate(frame.copy(), dets)
            out_frame = box_ann.annotate(out_frame, dets)
            out_frame = label_ann.annotate(out_frame, dets, labels=labels)
            if line_zone is not None:
                out_frame = line_ann.annotate(out_frame, line_counter=line_zone)
            sink.write_frame(out_frame)

    print("saved tracked video to:", video_out)
    if line_zone is not None:
        print("crossed in:", line_zone.in_count, "| crossed out:", line_zone.out_count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-time aerial human tracking using OpenVINO INT8 + ByteTrack + Supervision')
    parser.add_argument('--model', type=str, required=True, help='Path to OpenVINO INT8 folder, .onnx, or .pt model')
    parser.add_argument('--source', type=str, required=True, help='Input video path (.mp4)')
    parser.add_argument('--output', type=str, default='tracked_output.mp4', help='Output tracked video path (.mp4)')
    parser.add_argument('--imgsz', type=int, default=640, help='Inference resolution')
    parser.add_argument('--conf', type=float, default=0.25, help='Detection confidence threshold')
    parser.add_argument('--device', type=str, default='cpu', help='Inference device (cpu or 0)')
    parser.add_argument('--line', type=str, default='vertical', choices=['none', 'vertical', 'horizontal'], help='Optional LineZone counter')
    args = parser.parse_args()
    track_video(args.model, args.source, args.output, args.imgsz, args.conf, args.device, args.line)
