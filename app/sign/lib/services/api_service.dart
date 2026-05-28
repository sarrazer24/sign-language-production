import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

class ApiService {
  static const String baseUrl = 'https://asl-backend-s59i.onrender.com/api';

  // ── Token management ──
  static Future<String?> getToken() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString('token');
  }

  static Future<void> saveToken(String token) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('token', token);
  }

  static Future<void> saveUser(Map<String, dynamic> user) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('user', jsonEncode(user));
  }

  static Future<Map<String, dynamic>?> getUser() async {
    final prefs = await SharedPreferences.getInstance();
    final userStr = prefs.getString('user');
    if (userStr == null) return null;
    return jsonDecode(userStr);
  }

  static Future<void> logout() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove('token');
    await prefs.remove('user');
  }

  static Future<Map<String, String>> _headers({bool auth = false}) async {
    final headers = {'Content-Type': 'application/json'};
    if (auth) {
      final token = await getToken();
      if (token != null) headers['Authorization'] = 'Bearer $token';
    }
    return headers;
  }

  // ══════════════════════════════════════════
  //  AUTH
  // ══════════════════════════════════════════

  static Future<Map<String, dynamic>> signUp({
    required String fullName,
    required String email,
    required String password,
  }) async {
    final res = await http.post(
      Uri.parse('$baseUrl/auth/signup'),
      headers: await _headers(),
      body: jsonEncode({
        'full_name': fullName,
        'email': email,
        'password': password,
      }),
    );
    final data = jsonDecode(res.body);
    if (res.statusCode == 201) {
      await saveToken(data['token']);
      await saveUser(data['user']);
    }
    return {'status': res.statusCode, 'data': data};
  }

  static Future<Map<String, dynamic>> signIn({
    required String email,
    required String password,
  }) async {
    final res = await http.post(
      Uri.parse('$baseUrl/auth/signin'),
      headers: await _headers(),
      body: jsonEncode({'email': email, 'password': password}),
    );
    final data = jsonDecode(res.body);
    if (res.statusCode == 200) {
      await saveToken(data['token']);
      await saveUser(data['user']);
    }
    return {'status': res.statusCode, 'data': data};
  }

  static Future<Map<String, dynamic>> getProfile() async {
    final res = await http.get(
      Uri.parse('$baseUrl/auth/profile'),
      headers: await _headers(auth: true),
    );
    return {'status': res.statusCode, 'data': jsonDecode(res.body)};
  }

  static Future<Map<String, dynamic>> forgotPassword({
    required String email,
  }) async {
    final res = await http.post(
      Uri.parse('$baseUrl/auth/forgot-password'),
      headers: await _headers(),
      body: jsonEncode({'email': email}),
    );
    return {'status': res.statusCode, 'data': jsonDecode(res.body)};
  }

  // ══════════════════════════════════════════
  //  SIGNS (Dictionary)
  // ══════════════════════════════════════════

  static Future<List<dynamic>> getSigns(
      {String? category, String? search}) async {
    String url = '$baseUrl/signs';
    final params = <String>[];
    if (category != null) params.add('category=$category');
    if (search != null && search.isNotEmpty) params.add('search=$search');
    if (params.isNotEmpty) url += '?${params.join('&')}';

    final res = await http.get(
      Uri.parse(url),
      headers: await _headers(auth: true),
    );
    if (res.statusCode == 200) return jsonDecode(res.body);
    return [];
  }

  // ══════════════════════════════════════════
  //  GENERATIONS (AI Studio)
  // ══════════════════════════════════════════

  static Future<Map<String, dynamic>> createGeneration(String text) async {
    final res = await http.post(
      Uri.parse('$baseUrl/generations'),
      headers: await _headers(auth: true),
      body: jsonEncode({'original_text': text}),
    );
    return {'status': res.statusCode, 'data': jsonDecode(res.body)};
  }

  static Future<List<dynamic>> getMyGenerations() async {
    final res = await http.get(
      Uri.parse('$baseUrl/generations'),
      headers: await _headers(auth: true),
    );
    if (res.statusCode == 200) return jsonDecode(res.body);
    return [];
  }

  // ══════════════════════════════════════════
  //  ACTIVITY
  // ══════════════════════════════════════════

  static Future<List<dynamic>> getActivity() async {
    final res = await http.get(
      Uri.parse('$baseUrl/activity'),
      headers: await _headers(auth: true),
    );
    if (res.statusCode == 200) return jsonDecode(res.body);
    return [];
  }

  // ══════════════════════════════════════════
  //  NOTIFICATIONS
  // ══════════════════════════════════════════

  static Future<List<dynamic>> getNotifications() async {
    final res = await http.get(
      Uri.parse('$baseUrl/notifications'),
      headers: await _headers(auth: true),
    );
    if (res.statusCode == 200) return jsonDecode(res.body);
    return [];
  }

  static Future<void> clearNotifications() async {
    await http.delete(
      Uri.parse('$baseUrl/notifications/clear'),
      headers: await _headers(auth: true),
    );
  }
}
