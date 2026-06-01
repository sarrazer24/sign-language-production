import 'dart:async';
import 'dart:ui_web' as ui;

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:web/web.dart' as web;

import '../widgets/main_navigation.dart';

class SignToTextScreen extends StatefulWidget {
  const SignToTextScreen({super.key});

  @override
  State<SignToTextScreen> createState() => _SignToTextScreenState();
}

class _SignToTextScreenState extends State<SignToTextScreen> {
  static const String _apiBaseUrl = 'http://localhost:5000';
  static const String _apiErrorMessage =
      'API offline - run: python local_api.py';
  static const String _viewType = 'asl-mjpeg-view';

  static bool _viewRegistered = false;
  static web.HTMLImageElement? _registeredImageEl;

  late final http.Client _httpClient;
  web.HTMLImageElement? _imageEl;
  bool _isRecording = false;
  bool _apiOnline = true;
  int _failedRequests = 0;
  Timer? _healthTimer;

  @override
  void initState() {
    super.initState();
    _httpClient = http.Client();
    _initializeStreamElement();
    Future.delayed(const Duration(milliseconds: 300), _checkApiHealth);
    _healthTimer = Timer.periodic(
      const Duration(seconds: 10),
      (_) => _checkApiHealth(),
    );
  }

  void _initializeStreamElement() {
    _imageEl = web.HTMLImageElement()
      ..style.width = '100%'
      ..style.height = '100%'
      ..style.objectFit = 'cover'
      ..style.display = 'block';
    _registeredImageEl = _imageEl;

    if (!_viewRegistered) {
      ui.platformViewRegistry.registerViewFactory(
        _viewType,
        (int viewId) => _registeredImageEl ?? web.HTMLDivElement(),
      );
      _viewRegistered = true;
    }
  }

  @override
  void dispose() {
    _stopStream(updateState: false);
    _healthTimer?.cancel();
    _httpClient.close();
    super.dispose();
  }

  Future<void> _checkApiHealth() async {
    try {
      final response = await _httpClient
          .get(Uri.parse('$_apiBaseUrl/health'))
          .timeout(const Duration(seconds: 3));

      if (response.statusCode != 200) {
        throw Exception('Health check failed');
      }

      _markApiSuccess();
    } catch (_) {
      _markApiFailure();
    }
  }

  void _markApiSuccess() {
    final shouldRebuild = !_apiOnline || _failedRequests != 0;
    _apiOnline = true;
    _failedRequests = 0;

    if (shouldRebuild && mounted) {
      setState(() {});
    }
  }

  void _markApiFailure() {
    final wasShowingBanner = _showOfflineBanner;
    _failedRequests += 1;
    _apiOnline = false;
    final shouldRebuild = wasShowingBanner != _showOfflineBanner;

    if (shouldRebuild && mounted) {
      setState(() {});
    }
  }

  bool get _showOfflineBanner => !_apiOnline && _failedRequests >= 3;

  Future<void> _startStream() async {
    try {
      await _checkApiHealth();
      if (!_apiOnline) {
        throw Exception('API offline');
      }

      _imageEl?.src =
          '$_apiBaseUrl/video_feed?ts=${DateTime.now().millisecondsSinceEpoch}';

      if (!mounted) return;
      setState(() => _isRecording = true);
    } catch (_) {
      _markApiFailure();
      _showSnackBar(_apiErrorMessage);
    }
  }

  Future<void> _stopStream({bool updateState = true}) async {
    _imageEl?.src = '';

    try {
      await _httpClient
          .post(Uri.parse('$_apiBaseUrl/stop_stream'))
          .timeout(const Duration(seconds: 2));
    } catch (_) {
      // The image source is already detached, so failing to notify Flask is ok.
    }

    if (updateState && mounted) {
      setState(() => _isRecording = false);
    }
  }

  void _showSnackBar(String message) {
    if (!mounted) return;

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), backgroundColor: Colors.red),
    );
  }

  void _goBack() {
    _stopStream();
    Navigator.pushAndRemoveUntil(
      context,
      MaterialPageRoute(builder: (_) => const MainNavigation()),
      (route) => false,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF8F7FF),
      body: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const SizedBox(height: 24),
                  Row(
                    children: [
                      GestureDetector(
                        onTap: _goBack,
                        child: Container(
                          width: 36,
                          height: 36,
                          decoration: BoxDecoration(
                            color: const Color(0xFFF0EFF8),
                            borderRadius: BorderRadius.circular(10),
                          ),
                          child: const Icon(
                            Icons.arrow_back_ios_new_rounded,
                            size: 16,
                            color: Color(0xFF5B4FCF),
                          ),
                        ),
                      ),
                      const SizedBox(width: 12),
                      const Text(
                        'Sign to Text',
                        style: TextStyle(
                          fontSize: 26,
                          fontWeight: FontWeight.bold,
                          color: Color(0xFF1A1A2E),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 4),
                  const Padding(
                    padding: EdgeInsets.only(left: 48),
                    child: Text(
                      'Translate sign language to text with AI',
                      style: TextStyle(fontSize: 13, color: Colors.grey),
                    ),
                  ),
                  const SizedBox(height: 24),
                ],
              ),
            ),
            if (_showOfflineBanner) _buildOfflineBanner(),
            Expanded(
              child: _isRecording ? _buildStreamView() : _buildStartView(),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildOfflineBanner() {
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.fromLTRB(20, 0, 20, 12),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: const Color(0xFFFFF4D8),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: const Color(0xFFFFC857)),
      ),
      child: const Text(
        _apiErrorMessage,
        style: TextStyle(
          color: Color(0xFF7A4B00),
          fontSize: 13,
          fontWeight: FontWeight.w700,
        ),
      ),
    );
  }

  Widget _buildStreamView() {
    return Stack(
      fit: StackFit.expand,
      children: [
        const HtmlElementView(viewType: _viewType),
        Positioned(
          left: 72,
          right: 72,
          bottom: 18,
          child: GestureDetector(
            onTap: _stopStream,
            child: Container(
              padding: const EdgeInsets.symmetric(vertical: 16),
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  colors: [Color(0xFF7B6EF6), Color(0xFF5B4FCF)],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius: BorderRadius.circular(18),
                boxShadow: [
                  BoxShadow(
                    color: const Color(0xFF5B4FCF).withValues(alpha: 0.35),
                    blurRadius: 16,
                    offset: const Offset(0, 6),
                  ),
                ],
              ),
              child: const Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.stop_rounded, color: Colors.white, size: 22),
                  SizedBox(width: 10),
                  Text(
                    'Stop Recording',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildStartView() {
    return SingleChildScrollView(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Column(
        children: [
          const SizedBox(height: 136),
          GestureDetector(
            onTap: _startStream,
            child: Container(
              width: double.infinity,
              padding: const EdgeInsets.symmetric(vertical: 18),
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  colors: [Color(0xFF7B6EF6), Color(0xFF5B4FCF)],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius: BorderRadius.circular(18),
                boxShadow: [
                  BoxShadow(
                    color: const Color(0xFF5B4FCF).withValues(alpha: 0.35),
                    blurRadius: 16,
                    offset: const Offset(0, 6),
                  ),
                ],
              ),
              child: const Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.videocam_rounded, color: Colors.white, size: 22),
                  SizedBox(width: 10),
                  Text(
                    'Start Recording',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 32),
        ],
      ),
    );
  }
}
