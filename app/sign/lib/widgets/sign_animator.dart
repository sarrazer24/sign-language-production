import 'package:flutter/material.dart';
import 'pose_painter.dart';
import '../services/sign_service.dart';

class SignAnimator extends StatefulWidget {
  final SignResult result;
  final int fps;

  const SignAnimator({super.key, required this.result, this.fps = 24});

  @override
  State<SignAnimator> createState() => _SignAnimatorState();
}

class _SignAnimatorState extends State<SignAnimator>
    with SingleTickerProviderStateMixin {
  late AnimationController _ctrl;
  int _frame = 0;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      vsync: this,
      duration: Duration(
        milliseconds: (widget.result.nFrames / widget.fps * 1000).round(),
      ),
    )..addListener(() {
        setState(() {
          _frame = (_ctrl.value * (widget.result.nFrames - 1)).round();
        });
      })
      ..repeat();
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // ── Animated skeleton ─────────────────────────────────────────
        AspectRatio(
          aspectRatio: 0.75,
          child: Container(
            decoration: BoxDecoration(
              color: Colors.black87,
              borderRadius: BorderRadius.circular(16),
            ),
            child: CustomPaint(
              painter: PosePainter(widget.result.poses[_frame]),
            ),
          ),
        ),
        const SizedBox(height: 8),

        // ── Playback controls ─────────────────────────────────────────
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            IconButton(
              icon: Icon(_ctrl.isAnimating ? Icons.pause : Icons.play_arrow),
              onPressed: () => _ctrl.isAnimating ? _ctrl.stop() : _ctrl.repeat(),
            ),
            Text('Frame $_frame / ${widget.result.nFrames - 1}',
                style: const TextStyle(fontSize: 12)),
            IconButton(
              icon: const Icon(Icons.replay),
              onPressed: () => _ctrl
                ..reset()
                ..repeat(),
            ),
          ],
        ),

        // ── Raw pose data panel ───────────────────────────────────────
        ExpansionTile(
          title: const Text('Raw pose data', style: TextStyle(fontSize: 13)),
          children: [
            SizedBox(
              height: 150,
              child: ListView.builder(
                itemCount: widget.result.poses[_frame].length,
                itemBuilder: (ctx, i) {
                  final kp = widget.result.poses[_frame][i];
                  return Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 2),
                    child: Text(
                      'kp[$i]  x:${kp[0].toStringAsFixed(4)}'
                      '  y:${kp[1].toStringAsFixed(4)}'
                      '  c:${kp[2].toStringAsFixed(2)}',
                      style: const TextStyle(
                          fontFamily: 'monospace', fontSize: 11),
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ],
    );
  }
}