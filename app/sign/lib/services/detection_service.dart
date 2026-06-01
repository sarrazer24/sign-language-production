import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;

class DetectionService {
  static const String _baseUrl = 'http://localhost:5000';

  /// Send image bytes to the API and return the top detected sign.
  /// Returns null if nothing detected or on error.
  static Future<String?> predict(Uint8List imageBytes) async {
    try {
      final base64Image = base64Encode(imageBytes);
      final response = await http
          .post(
            Uri.parse('$_baseUrl/predict_base64'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({'image': base64Image}),
          )
          .timeout(const Duration(seconds: 10));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data['success'] == true && data['top_prediction'] != null) {
          return data['top_prediction'] as String;
        }
      }
      return null;
    } catch (_) {
      return null;
    }
  }
}