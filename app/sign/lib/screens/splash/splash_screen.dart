import 'package:flutter/material.dart';
import '../../widgets/splash_slide.dart';
import '../auth/sign_in_screen.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  final PageController _pageController = PageController();
  int _currentIndex = 0;

  static final List<Map<String, String>> slides = [
    {
      'image': 'assets/images/splash_1.png',
      'title': 'Type Any Text',
      'description':
          'Enter any text you want to convert to sign language. Our AI understands context and nuance.',
    },
    {
      'image': 'assets/images/splash_2.png',
      'title': 'Or Speak Your Message',
      'description':
          "Can't type? No problem! Record a voice message and our AI will transcribe and convert it to sign language instantly.",
    },
    {
      'image': 'assets/images/splash_3.png',
      'title': 'AI Pipeline Magic',
      'description':
          'Our two-stage AI converts text to poses, then generates realistic sign language videos using advanced diffusion technology.',
    },
    {
      'image': 'assets/images/splash_4.png',
      'title': 'Watch & Share',
      'description':
          'View high-quality sign language videos instantly. Download or share them with anyone who needs them.',
    },
    {
      'image': 'assets/images/splash_5.png',
      'title': 'Help Us Improve',
      'description':
          'Your feedback trains our AI to be more accurate. Together, we make communication accessible for everyone.',
    },
  ];

  void _goToSignIn() {
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (_) => const SignInScreen()),
    );
  }

  void _nextPage() {
    if (_currentIndex < slides.length - 1) {
      _pageController.nextPage(
        duration: const Duration(milliseconds: 400),
        curve: Curves.easeInOut,
      );
    } else {
      _goToSignIn();
    }
  }

  @override
  void dispose() {
    _pageController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: Column(
        children: [
          // ── Page content ──
          Expanded(
            child: PageView.builder(
              controller: _pageController,
              itemCount: slides.length,
              onPageChanged: (index) => setState(() => _currentIndex = index),
              itemBuilder: (context, index) {
                final slide = slides[index];
                return SplashSlide(
                  image: slide['image']!,
                  title: slide['title']!,
                  description: slide['description']!,
                );
              },
            ),
          ),

          // ── Dot indicators ──
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: List.generate(slides.length, (index) {
              final isActive = index == _currentIndex;
              return AnimatedContainer(
                duration: const Duration(milliseconds: 300),
                margin: const EdgeInsets.symmetric(horizontal: 4),
                width: isActive ? 24 : 8,
                height: 8,
                decoration: BoxDecoration(
                  color: isActive
                      ? const Color(0xFF5B4FCF)
                      : const Color(0xFFD0CDF7),
                  borderRadius: BorderRadius.circular(4),
                ),
              );
            }),
          ),

          const SizedBox(height: 32),

          // ── Skip + Next row ──
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 32.0),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                // Skip button
                TextButton(
                  onPressed: _goToSignIn,
                  style: TextButton.styleFrom(
                    backgroundColor: const Color(0xFFF0EFF8),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(30),
                    ),
                    padding: const EdgeInsets.symmetric(
                        horizontal: 24, vertical: 12),
                  ),
                  child: const Text(
                    'Skip',
                    style: TextStyle(
                      color: Color(0xFF5B4FCF),
                      fontWeight: FontWeight.w500,
                      fontSize: 15,
                    ),
                  ),
                ),

                // Next button
                GestureDetector(
                  onTap: _nextPage,
                  child: Container(
                    width: 52,
                    height: 52,
                    decoration: const BoxDecoration(
                      shape: BoxShape.circle,
                      color: Color(0xFF5B4FCF),
                    ),
                    child: const Icon(
                      Icons.arrow_forward_ios_rounded,
                      color: Colors.white,
                      size: 20,
                    ),
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(height: 40),
        ],
      ),
    );
  }
}