import 'package:flutter/material.dart';

class SplashSlide extends StatelessWidget {
  final String image;       // 'assets/images/splash_1.png'
  final String title;
  final String description;

  const SplashSlide({
    super.key,
    required this.image,
    required this.title,
    required this.description,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 40.0),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          // ── Circle image container ──
          Container(
            width: 160,
            height: 160,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              gradient: const LinearGradient(
                colors: [Color(0xFF7B6EF6), Color(0xFF5B4FCF)],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
              boxShadow: [
                BoxShadow(
                  color: const Color(0xFF5B4FCF).withOpacity(0.25),
                  blurRadius: 24,
                  offset: const Offset(0, 8),
                ),
              ],
            ),
            child: ClipOval(
              child: Image.asset(
                image,
                fit: BoxFit.cover,
                errorBuilder: (context, error, stackTrace) {
                  // fallback si l'image est absente
                  return const Icon(
                    Icons.image_not_supported_outlined,
                    color: Colors.white,
                    size: 60,
                  );
                },
              ),
            ),
          ),

          const SizedBox(height: 48),

          // ── Title ──
          Text(
            title,
            style: const TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.bold,
              color: Color(0xFF1A1A2E),
            ),
            textAlign: TextAlign.center,
          ),

          const SizedBox(height: 16),

          // ── Description ──
          Text(
            description,
            style: const TextStyle(
              fontSize: 14,
              color: Colors.grey,
              height: 1.6,
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }
}