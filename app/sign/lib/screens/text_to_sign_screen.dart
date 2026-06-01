import 'package:flutter/material.dart';
import '../widgets/main_navigation.dart';
import 'generated_video_screen.dart';

class TextToSignScreen extends StatefulWidget {
  const TextToSignScreen({super.key});

  @override
  State<TextToSignScreen> createState() => _TextToSignScreenState();
}

class _TextToSignScreenState extends State<TextToSignScreen> {
  // 'home' | 'text' | 'voice'
  String _currentPage = 'home';

  final TextEditingController _textController = TextEditingController();
  bool _isRecording = false;
  bool _isProcessing = false;

  @override
  void dispose() {
    _textController.dispose();
    super.dispose();
  }

  void _goBack() {
    if (_currentPage != 'home') {
      setState(() {
        _currentPage = 'home';
        _isRecording = false;
        _isProcessing = false;
        _textController.clear();
      });
    } else {
      Navigator.pushAndRemoveUntil(
        context,
        MaterialPageRoute(builder: (_) => const MainNavigation()),
        (route) => false,
      );
    }
  }

  void _toggleRecording() {
    setState(() => _isRecording = !_isRecording);
    if (!_isRecording) {
      setState(() => _isProcessing = true);
      Future.delayed(const Duration(seconds: 2), () {
        if (mounted) {
          setState(() {
            _isProcessing = false;
            _textController.text = 'Hello, how are you?';
          });
        }
      });
    } else {
      _textController.clear();
    }
  }

  void _translateToSign() {
    final text = _textController.text.trim();
    if (text.isEmpty) return;
    setState(() => _isProcessing = true);
    Future.delayed(const Duration(seconds: 2), () {
      if (mounted) {
        setState(() => _isProcessing = false);
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (_) => GeneratedVideoScreen(originalText: text),
          ),
        );
      }
    });
  }

  // ─────────────────────────────────────────
  // HEADER commun
  // ─────────────────────────────────────────
  Widget _buildHeader(String subtitle) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 24, 20, 0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
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
                  child: const Icon(Icons.arrow_back_ios_new_rounded,
                      size: 16, color: Color(0xFF5B4FCF)),
                ),
              ),
              const SizedBox(width: 12),
              const Text('Text to Sign',
                  style: TextStyle(
                      fontSize: 26,
                      fontWeight: FontWeight.bold,
                      color: Color(0xFF1A1A2E))),
            ],
          ),
          const SizedBox(height: 4),
          Padding(
            padding: const EdgeInsets.only(left: 48),
            child: Text(subtitle,
                style: const TextStyle(fontSize: 13, color: Colors.grey)),
          ),
        ],
      ),
    );
  }

  // ─────────────────────────────────────────
  // PAGE HOME — choisir le mode
  // ─────────────────────────────────────────
  Widget _buildHomePage() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildHeader('Choose how you want to input'),
        const SizedBox(height: 40),
        Expanded(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: Column(
              children: [
                // Type Text card
                GestureDetector(
                  onTap: () => setState(() => _currentPage = 'text'),
                  child: Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(24),
                    decoration: BoxDecoration(
                      gradient: const LinearGradient(
                        colors: [Color(0xFF7B6EF6), Color(0xFF5B4FCF)],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                      ),
                      borderRadius: BorderRadius.circular(22),
                      boxShadow: [
                        BoxShadow(
                          color:
                              const Color(0xFF5B4FCF).withValues(alpha: 0.35),
                          blurRadius: 20,
                          offset: const Offset(0, 8),
                        )
                      ],
                    ),
                    child: Row(
                      children: [
                        Container(
                          width: 56,
                          height: 56,
                          decoration: BoxDecoration(
                            color: Colors.white.withValues(alpha: 0.2),
                            borderRadius: BorderRadius.circular(16),
                          ),
                          child: const Icon(Icons.keyboard_rounded,
                              color: Colors.white, size: 30),
                        ),
                        const SizedBox(width: 18),
                        const Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text('Type a Message',
                                  style: TextStyle(
                                      fontSize: 18,
                                      fontWeight: FontWeight.bold,
                                      color: Colors.white)),
                              SizedBox(height: 6),
                              Text('Write your text manually',
                                  style: TextStyle(
                                      fontSize: 13, color: Colors.white70)),
                            ],
                          ),
                        ),
                        Container(
                          width: 36,
                          height: 36,
                          decoration: BoxDecoration(
                            color: Colors.white.withValues(alpha: 0.2),
                            shape: BoxShape.circle,
                          ),
                          child: const Icon(Icons.arrow_forward_rounded,
                              color: Colors.white, size: 18),
                        ),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 16),

                // Voice card
                GestureDetector(
                  onTap: () => setState(() => _currentPage = 'voice'),
                  child: Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(24),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(22),
                      border: Border.all(
                          color: const Color(0xFFE0DEFF), width: 1.5),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withValues(alpha: 0.05),
                          blurRadius: 14,
                          offset: const Offset(0, 4),
                        )
                      ],
                    ),
                    child: Row(
                      children: [
                        Container(
                          width: 56,
                          height: 56,
                          decoration: BoxDecoration(
                            color: const Color(0xFFF0EFF8),
                            borderRadius: BorderRadius.circular(16),
                          ),
                          child: const Icon(Icons.mic_rounded,
                              color: Color(0xFF5B4FCF), size: 30),
                        ),
                        const SizedBox(width: 18),
                        const Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text('Voice Recording',
                                  style: TextStyle(
                                      fontSize: 18,
                                      fontWeight: FontWeight.bold,
                                      color: Color(0xFF1A1A2E))),
                              SizedBox(height: 6),
                              Text('Speak and let AI transcribe',
                                  style: TextStyle(
                                      fontSize: 13, color: Colors.grey)),
                            ],
                          ),
                        ),
                        Container(
                          width: 36,
                          height: 36,
                          decoration: const BoxDecoration(
                            color: Color(0xFFF0EFF8),
                            shape: BoxShape.circle,
                          ),
                          child: const Icon(Icons.arrow_forward_rounded,
                              color: Color(0xFF5B4FCF), size: 18),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  // ─────────────────────────────────────────
  // PAGE TEXT
  // ─────────────────────────────────────────
  Widget _buildTextPage() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildHeader('Type your message below'),
        const SizedBox(height: 28),
        Expanded(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: Column(
              children: [
                // Text field
                Expanded(
                  child: Container(
                    width: double.infinity,
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(20),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withValues(alpha: 0.05),
                          blurRadius: 14,
                          offset: const Offset(0, 4),
                        )
                      ],
                    ),
                    padding: const EdgeInsets.all(18),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Row(
                          children: [
                            Icon(Icons.edit_rounded,
                                size: 14, color: Color(0xFF5B4FCF)),
                            SizedBox(width: 6),
                            Text('Your message',
                                style: TextStyle(
                                    fontSize: 12,
                                    fontWeight: FontWeight.w600,
                                    color: Color(0xFF5B4FCF))),
                          ],
                        ),
                        const SizedBox(height: 12),
                        Expanded(
                          child: TextField(
                            controller: _textController,
                            maxLines: null,
                            expands: true,
                            autofocus: true,
                            textAlignVertical: TextAlignVertical.top,
                            style: const TextStyle(
                              fontSize: 17,
                              color: Color(0xFF1A1A2E),
                              height: 1.6,
                            ),
                            decoration: const InputDecoration(
                              hintText: 'Type here...',
                              hintStyle:
                                  TextStyle(color: Colors.grey, fontSize: 17),
                              border: InputBorder.none,
                            ),
                          ),
                        ),
                        const Divider(color: Color(0xFFF0EFF8)),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            GestureDetector(
                              onTap: () => _textController.clear(),
                              child: const Row(
                                children: [
                                  Icon(Icons.close_rounded,
                                      size: 14, color: Colors.grey),
                                  SizedBox(width: 4),
                                  Text('Clear',
                                      style: TextStyle(
                                          fontSize: 12, color: Colors.grey)),
                                ],
                              ),
                            ),
                            ValueListenableBuilder<TextEditingValue>(
                              valueListenable: _textController,
                              builder: (_, v, __) => Text(
                                '${v.text.length} chars',
                                style: const TextStyle(
                                    fontSize: 11, color: Colors.grey),
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 20),

                // Translate button
                GestureDetector(
                  onTap: _isProcessing ? null : _translateToSign,
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
                              const Color(0xFF5B4FCF).withValues(alpha: 0.35),
                          blurRadius: 16,
                          offset: const Offset(0, 6),
                        )
                      ],
                    ),
                    child: _isProcessing
                        ? const Center(
                            child: SizedBox(
                              width: 24,
                              height: 24,
                              child: CircularProgressIndicator(
                                  color: Colors.white, strokeWidth: 2.5),
                            ),
                          )
                        : const Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Icon(Icons.sign_language_rounded,
                                  color: Colors.white, size: 22),
                              SizedBox(width: 10),
                              Text('Translate to Sign',
                                  style: TextStyle(
                                      fontSize: 16,
                                      fontWeight: FontWeight.bold,
                                      color: Colors.white)),
                            ],
                          ),
                  ),
                ),
                const SizedBox(height: 24),
              ],
            ),
          ),
        ),
      ],
    );
  }

  // ─────────────────────────────────────────
  // PAGE VOICE
  // ─────────────────────────────────────────
  Widget _buildVoicePage() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildHeader('Record your voice message'),
        const SizedBox(height: 40),
        Expanded(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: Column(
              children: [
                // Mic area
                Expanded(
                  child: Container(
                    width: double.infinity,
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(22),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withValues(alpha: 0.05),
                          blurRadius: 14,
                          offset: const Offset(0, 4),
                        )
                      ],
                    ),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        // Mic button with ripple effect
                        GestureDetector(
                          onTap: _toggleRecording,
                          child: Stack(
                            alignment: Alignment.center,
                            children: [
                              if (_isRecording) ...[
                                _buildRipple(
                                    140,
                                    (_isRecording
                                            ? Colors.red
                                            : const Color(0xFF5B4FCF))
                                        .withValues(alpha: 0.08)),
                                _buildRipple(
                                    110,
                                    (_isRecording
                                            ? Colors.red
                                            : const Color(0xFF5B4FCF))
                                        .withValues(alpha: 0.12)),
                              ],
                              Container(
                                width: 88,
                                height: 88,
                                decoration: BoxDecoration(
                                  shape: BoxShape.circle,
                                  color: _isRecording
                                      ? Colors.red
                                      : const Color(0xFF5B4FCF),
                                  boxShadow: [
                                    BoxShadow(
                                      color: (_isRecording
                                              ? Colors.red
                                              : const Color(0xFF5B4FCF))
                                          .withValues(alpha: 0.4),
                                      blurRadius: 24,
                                      offset: const Offset(0, 8),
                                    )
                                  ],
                                ),
                                child: Icon(
                                  _isRecording
                                      ? Icons.stop_rounded
                                      : Icons.mic_rounded,
                                  color: Colors.white,
                                  size: 40,
                                ),
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(height: 24),
                        Text(
                          _isRecording
                              ? 'Recording...'
                              : 'Tap to start recording',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w600,
                            color: _isRecording
                                ? Colors.red
                                : const Color(0xFF1A1A2E),
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          _isRecording
                              ? 'Tap again to stop'
                              : 'Your voice will be converted to text',
                          style:
                              const TextStyle(fontSize: 13, color: Colors.grey),
                        ),

                        if (_isProcessing && !_isRecording) ...[
                          const SizedBox(height: 28),
                          const CircularProgressIndicator(
                              color: Color(0xFF5B4FCF)),
                          const SizedBox(height: 10),
                          const Text('Transcribing your voice...',
                              style:
                                  TextStyle(fontSize: 13, color: Colors.grey)),
                        ],

                        // Result text
                        if (!_isProcessing &&
                            !_isRecording &&
                            _textController.text.isNotEmpty) ...[
                          const SizedBox(height: 28),
                          Padding(
                            padding: const EdgeInsets.symmetric(horizontal: 24),
                            child: Container(
                              width: double.infinity,
                              padding: const EdgeInsets.all(16),
                              decoration: BoxDecoration(
                                color: const Color(0xFFF0EFF8),
                                borderRadius: BorderRadius.circular(14),
                              ),
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  const Row(
                                    children: [
                                      Icon(Icons.record_voice_over_rounded,
                                          size: 14, color: Color(0xFF5B4FCF)),
                                      SizedBox(width: 6),
                                      Text('Transcribed',
                                          style: TextStyle(
                                              fontSize: 11,
                                              fontWeight: FontWeight.w600,
                                              color: Color(0xFF5B4FCF))),
                                    ],
                                  ),
                                  const SizedBox(height: 8),
                                  Text(
                                    _textController.text,
                                    style: const TextStyle(
                                        fontSize: 15,
                                        color: Color(0xFF1A1A2E),
                                        height: 1.5),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ],
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 20),

                // Translate button
                if (!_isRecording &&
                    !_isProcessing &&
                    _textController.text.isNotEmpty)
                  GestureDetector(
                    onTap: _translateToSign,
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
                                const Color(0xFF5B4FCF).withValues(alpha: 0.35),
                            blurRadius: 16,
                            offset: const Offset(0, 6),
                          )
                        ],
                      ),
                      child: const Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(Icons.sign_language_rounded,
                              color: Colors.white, size: 22),
                          SizedBox(width: 10),
                          Text('Translate to Sign',
                              style: TextStyle(
                                  fontSize: 16,
                                  fontWeight: FontWeight.bold,
                                  color: Colors.white)),
                        ],
                      ),
                    ),
                  ),
                const SizedBox(height: 24),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildRipple(double size, Color color) {
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(shape: BoxShape.circle, color: color),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF8F7FF),
      resizeToAvoidBottomInset: true,
      body: SafeArea(
        child: AnimatedSwitcher(
          duration: const Duration(milliseconds: 250),
          child: _currentPage == 'home'
              ? _buildHomePage()
              : _currentPage == 'text'
                  ? _buildTextPage()
                  : _buildVoicePage(),
        ),
      ),
    );
  }
}
