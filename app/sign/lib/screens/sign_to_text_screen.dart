import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import '../widgets/main_navigation.dart';
import '../services/api_service.dart';

class SignToTextScreen extends StatefulWidget {
  const SignToTextScreen({super.key});

  @override
  State<SignToTextScreen> createState() => _SignToTextScreenState();
}

class _SignToTextScreenState extends State<SignToTextScreen> {
  CameraController? _cameraController;
  bool _isRecording = false;
  bool _cameraReady = false;
  String _resultText = '';
  String? _errorText;

  @override
  void dispose() {
    _cameraController?.dispose();
    super.dispose();
  }

  Future<void> _startCamera() async {
    setState(() => _errorText = null);

    final cameras = await availableCameras();
    if (cameras.isEmpty) {
      setState(() => _errorText = 'Aucune caméra disponible.');
      return;
    }

    _cameraController = CameraController(
      cameras.first,
      ResolutionPreset.high,
      enableAudio: false,
    );

    await _cameraController!.initialize();

    // CORRECTION : on démarre vraiment l'enregistrement vidéo
    await _cameraController!.startVideoRecording();

    if (mounted) {
      setState(() {
        _cameraReady = true;
        _isRecording = true;
        _resultText = '';
      });
    }
  }

  Future<void> _stopCamera() async {
    XFile? videoFile;

    if (_cameraController != null && _isRecording) {
      //  Arrête l'enregistrement et récupère le fichier vidéo
      videoFile = await _cameraController!.stopVideoRecording();
    }

    await _cameraController?.dispose();
    _cameraController = null;

    if (mounted) {
      setState(() {
        _cameraReady = false;
        _isRecording = false;
        _resultText = 'Analyse en cours...';
        _errorText = null;
      });
    }

    if (videoFile != null) {
      // ✅ Envoie la vidéo au backend
      final result = await ApiService.analyzeVideo(videoFile.path);

      if (mounted) {
        if (result['status'] == 200) {
          setState(() {
            _resultText = result['data']['text'] ?? 'Aucun résultat';
          });
        } else {
          setState(() {
            _resultText = '';
            _errorText =
                result['data']['message'] ?? 'Erreur lors de l\'analyse.';
          });
        }
      }
    } else {
      if (mounted) {
        setState(() {
          _resultText = '';
          _errorText = 'Aucune vidéo enregistrée.';
        });
      }
    }
  }

  void _toggleRecording() {
    if (_isRecording) {
      _stopCamera();
    } else {
      _startCamera();
    }
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
                        onTap: () => Navigator.pushAndRemoveUntil(
                          context,
                          MaterialPageRoute(
                              builder: (_) => const MainNavigation()),
                          (route) => false,
                        ),
                        child: Container(
                          width: 36,
                          height: 36,
                          decoration: BoxDecoration(
                            color: const Color(0xFFF0EFF8),
                            borderRadius: BorderRadius.circular(10),
                          ),
                          child: const Icon(Icons.arrow_back_ios_new_rounded,
                              size: 16, color: Color(0xFF5B4FCF)),
                        ),
                      ),
                      const SizedBox(width: 12),
                      const Text('Sign to Text',
                          style: TextStyle(
                              fontSize: 26,
                              fontWeight: FontWeight.bold,
                              color: Color(0xFF1A1A2E))),
                    ],
                  ),
                  const SizedBox(height: 4),
                  const Padding(
                    padding: EdgeInsets.only(left: 48),
                    child: Text('Translate sign language to text with AI',
                        style: TextStyle(fontSize: 13, color: Colors.grey)),
                  ),
                  const SizedBox(height: 24),
                ],
              ),
            ),

            // Caméra active
            if (_cameraReady && _cameraController != null)
              Expanded(
                child: Stack(
                  children: [
                    CameraPreview(_cameraController!),
                    Positioned(
                      top: 16,
                      right: 16,
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 10, vertical: 4),
                        decoration: BoxDecoration(
                          color: Colors.red,
                          borderRadius: BorderRadius.circular(20),
                        ),
                        child: const Row(
                          children: [
                            Icon(Icons.circle, color: Colors.white, size: 8),
                            SizedBox(width: 4),
                            Text('LIVE',
                                style: TextStyle(
                                    color: Colors.white,
                                    fontSize: 11,
                                    fontWeight: FontWeight.bold)),
                          ],
                        ),
                      ),
                    ),
                    Positioned(
                      bottom: 30,
                      left: 0,
                      right: 0,
                      child: Center(
                        child: GestureDetector(
                          onTap: _toggleRecording,
                          child: Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 32, vertical: 16),
                            decoration: BoxDecoration(
                              color: Colors.red,
                              borderRadius: BorderRadius.circular(18),
                              boxShadow: [
                                BoxShadow(
                                    color: Colors.red.withOpacity(0.4),
                                    blurRadius: 16,
                                    offset: const Offset(0, 6))
                              ],
                            ),
                            child: const Row(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Icon(Icons.stop_rounded,
                                    color: Colors.white, size: 22),
                                SizedBox(width: 10),
                                Text('Stop Recording',
                                    style: TextStyle(
                                        fontSize: 16,
                                        fontWeight: FontWeight.bold,
                                        color: Colors.white)),
                              ],
                            ),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),

            // Caméra inactive
            if (!_isRecording)
              Expanded(
                child: SingleChildScrollView(
                  padding: const EdgeInsets.symmetric(horizontal: 20),
                  child: Column(
                    children: [
                      GestureDetector(
                        onTap: _toggleRecording,
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
                                  color:
                                      const Color(0xFF5B4FCF).withOpacity(0.35),
                                  blurRadius: 16,
                                  offset: const Offset(0, 6))
                            ],
                          ),
                          child: const Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Icon(Icons.videocam_rounded,
                                  color: Colors.white, size: 22),
                              SizedBox(width: 10),
                              Text('Start Recording',
                                  style: TextStyle(
                                      fontSize: 16,
                                      fontWeight: FontWeight.bold,
                                      color: Colors.white)),
                            ],
                          ),
                        ),
                      ),

                      // Message d'erreur
                      if (_errorText != null) ...[
                        const SizedBox(height: 24),
                        Container(
                          width: double.infinity,
                          padding: const EdgeInsets.all(16),
                          decoration: BoxDecoration(
                            color: Colors.red.shade50,
                            borderRadius: BorderRadius.circular(14),
                            border: Border.all(color: Colors.red.shade200),
                          ),
                          child: Row(
                            children: [
                              Icon(Icons.error_outline,
                                  color: Colors.red.shade400, size: 20),
                              const SizedBox(width: 10),
                              Expanded(
                                child: Text(_errorText!,
                                    style: TextStyle(
                                        color: Colors.red.shade700,
                                        fontSize: 14)),
                              ),
                            ],
                          ),
                        ),
                      ],

                      // Résultat
                      if (_resultText.isNotEmpty) ...[
                        const SizedBox(height: 24),
                        Container(
                          width: double.infinity,
                          padding: const EdgeInsets.all(20),
                          decoration: BoxDecoration(
                            color: Colors.white,
                            borderRadius: BorderRadius.circular(20),
                            boxShadow: [
                              BoxShadow(
                                  color: Colors.black.withOpacity(0.04),
                                  blurRadius: 12,
                                  offset: const Offset(0, 4))
                            ],
                          ),
                          child: _resultText == 'Analyse en cours...'
                              ? const Row(
                                  children: [
                                    SizedBox(
                                      width: 18,
                                      height: 18,
                                      child: CircularProgressIndicator(
                                          strokeWidth: 2,
                                          color: Color(0xFF5B4FCF)),
                                    ),
                                    SizedBox(width: 12),
                                    Text('Analyse en cours...',
                                        style: TextStyle(
                                            fontSize: 15, color: Colors.grey)),
                                  ],
                                )
                              : Text(
                                  _resultText,
                                  style: const TextStyle(
                                      fontSize: 18,
                                      color: Color(0xFF1A1A2E),
                                      height: 1.5),
                                ),
                        ),
                      ],

                      const SizedBox(height: 32),
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
