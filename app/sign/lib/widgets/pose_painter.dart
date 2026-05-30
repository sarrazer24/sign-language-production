import 'package:flutter/material.dart';

// OpenPose 25-point body connections
const List<List<int>> _bodyConnections = [
  [1,2],[1,5],[2,3],[3,4],[5,6],[6,7],
  [1,8],[8,9],[9,10],[1,11],[11,12],[12,13],
  [1,0],[0,14],[14,16],[0,15],[15,17],
];

const List<List<int>> _handConnections = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
];

class PosePainter extends CustomPainter {
  final List<List<double>> frame; // 151 keypoints, each [x, y, conf]

  PosePainter(this.frame);

  @override
  void paint(Canvas canvas, Size size) {
    final bodyPaint = Paint()
      ..color = Colors.cyanAccent
      ..strokeWidth = 2.5
      ..strokeCap = StrokeCap.round;

    final handPaint = Paint()
      ..color = Colors.greenAccent
      ..strokeWidth = 2.0
      ..strokeCap = StrokeCap.round;

    final dotPaint = Paint()..color = Colors.white;

    void drawConnections(
      List<List<int>> conns,
      int offset,
      Paint paint,
    ) {
      for (final c in conns) {
        final a = frame[offset + c[0]];
        final b = frame[offset + c[1]];
        if (a[2] < 0.1 || b[2] < 0.1) continue;
        canvas.drawLine(
          Offset(a[0] * size.width,  a[1] * size.height),
          Offset(b[0] * size.width,  b[1] * size.height),
          paint,
        );
      }
    }

    // Body (0–24)
    drawConnections(_bodyConnections, 0, bodyPaint);
    // Left hand (92–112)
    drawConnections(_handConnections, 92, handPaint);
    // Right hand (113–133)
    drawConnections(_handConnections, 113, handPaint);

    // Draw dots for all keypoints
    for (final kp in frame) {
      if (kp[2] < 0.1) continue;
      canvas.drawCircle(
        Offset(kp[0] * size.width, kp[1] * size.height),
        3.0,
        dotPaint,
      );
    }
  }

  @override
  bool shouldRepaint(PosePainter old) => old.frame != frame;
}