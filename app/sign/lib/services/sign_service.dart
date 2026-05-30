import 'dart:convert';
import 'package:http/http.dart' as http;

class SignService {
  // Change to your Node.js base URL
  static const String _base = 'http://YOUR_NODE_SERVER/api/signs';

  static Future<SignResult> generate({
    required String text,
    int nFrames = 60,
    double guidanceScale = 3.0,
  }) async {
    final res = await http.post(
      Uri.parse('$_base/generate'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'text': text,
        'n_frames': nFrames,
        'guidance_scale': guidanceScale,
      }),
    ).timeout(const Duration(seconds: 90));

    if (res.statusCode != 200) {
      throw Exception('Server error ${res.statusCode}: ${res.body}');
    }
    return SignResult.fromJson(jsonDecode(res.body));
  }
}

// ── Data models ──────────────────────────────────────────────────────────────

class SignResult {
  final int nFrames;
  final int nKeypoints;
  // poses[frame][keypoint] = [x, y, confidence]
  final List<List<List<double>>> poses;

  SignResult({
    required this.nFrames,
    required this.nKeypoints,
    required this.poses,
  });

  factory SignResult.fromJson(Map<String, dynamic> json) {
    final raw = json['poses'] as List;
    final poses = raw.map((frame) {
      return (frame as List).map((kp) {
        return (kp as List).map((v) => (v as num).toDouble()).toList();
      }).toList();
    }).toList();

    return SignResult(
      nFrames:    json['n_frames'],
      nKeypoints: json['n_keypoints'],
      poses:      poses,
    );
  }
}