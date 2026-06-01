import 'package:flutter/material.dart';

class SignCard extends StatelessWidget {
  final String letter;
  final String? imagePath;
  final VoidCallback? onTap;

  const SignCard({
    super.key,
    required this.letter,
    this.imagePath,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: const Color(0xFF5B4FCF)),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.04),
              blurRadius: 6,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Expanded(
              child: Padding(
                padding: const EdgeInsets.all(8.0),
                child: imagePath != null
                    ? Image.asset(
                        imagePath!,
                        fit: BoxFit.contain,
                        errorBuilder: (_, __, ___) => const Icon(
                          Icons.back_hand_outlined,
                          size: 40,
                          color: Color(0xFF5B4FCF),
                        ),
                      )
                    : const Icon(
                        Icons.back_hand_outlined,
                        size: 40,
                        color: Color(0xFF5B4FCF),
                      ),
              ),
            ),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.symmetric(vertical: 6),
              decoration: const BoxDecoration(
                color: Color(0xFFF8F7FF),
                borderRadius: BorderRadius.vertical(
                  bottom: Radius.circular(14),
                ),
              ),
              child: Text(
                letter,
                textAlign: TextAlign.center,
                style: const TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                  color: Color(0xFF1A1A2E),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
