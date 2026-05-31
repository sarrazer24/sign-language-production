import 'package:flutter/material.dart';
import '../widgets/custom_button.dart';
import '../services/api_service.dart';

class GeneratedVideoScreen extends StatelessWidget {
  final String originalText;
  final String? generationId;
  //  AJOUT : URL de la vidéo générée par le backend
  final String? videoUrl;

  const GeneratedVideoScreen({
    super.key,
    required this.originalText,
    this.generationId,
    this.videoUrl, //  reçu depuis TextToSignScreen
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 12, 12, 0),
              child: GestureDetector(
                onTap: () => Navigator.pop(context),
                child: Container(
                  width: 36,
                  height: 36,
                  decoration: BoxDecoration(
                      color: const Color(0xFFF0EFF8),
                      borderRadius: BorderRadius.circular(10)),
                  child: const Icon(Icons.arrow_back_ios_new_rounded,
                      size: 16, color: Color(0xFF5B4FCF)),
                ),
              ),
            ),
            const SizedBox(height: 12),

            // Zone vidéo
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Container(
                width: double.infinity,
                height: 220,
                decoration: BoxDecoration(
                    color: const Color(0xFFE8E8E8),
                    borderRadius: BorderRadius.circular(18)),
                child: videoUrl != null
                    //  Si on a une URL, on affiche un bouton "ouvrir la vidéo"
                    // (pour lire la vidéo tu peux intégrer video_player si besoin)
                    ? Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          const Icon(Icons.play_circle_filled_rounded,
                              size: 56, color: Color(0xFF5B4FCF)),
                          const SizedBox(height: 8),
                          Text(
                            'Vidéo prête',
                            style: TextStyle(
                                color: Colors.grey.shade600, fontSize: 13),
                          ),
                        ],
                      )
                    : const Center(
                        child: Icon(Icons.play_circle_outline_rounded,
                            size: 56, color: Color(0xFFBBBBBB)),
                      ),
              ),
            ),

            const SizedBox(height: 20),

            // Texte original
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: const Color(0xFFF0EFF8)),
                  boxShadow: [
                    BoxShadow(
                        color: Colors.black.withOpacity(0.04),
                        blurRadius: 8,
                        offset: const Offset(0, 2))
                  ],
                ),
                child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text('Original text:',
                          style: TextStyle(fontSize: 12, color: Colors.grey)),
                      const SizedBox(height: 4),
                      Text('"$originalText"',
                          style: const TextStyle(
                              fontSize: 15,
                              fontWeight: FontWeight.w500,
                              color: Color(0xFF1A1A2E))),
                    ]),
              ),
            ),

            const SizedBox(height: 20),

            // Actions
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Row(children: [
                _ActionButton(
                    icon: Icons.replay_rounded, label: 'Replay', onTap: () {}),
                const SizedBox(width: 12),
                _ActionButton(
                    icon: Icons.download_outlined,
                    label: 'Download',
                    onTap: () {}),
                const SizedBox(width: 12),
                _ActionButton(
                    icon: Icons.share_outlined, label: 'Share', onTap: () {}),
              ]),
            ),

            const SizedBox(height: 24),

            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: CustomButton(
                text: 'Rate This Video',
                onPressed: () => _showRatingSheet(context),
              ),
            ),
            const SizedBox(height: 12),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: SizedBox(
                width: double.infinity,
                height: 52,
                child: OutlinedButton(
                  onPressed: () => Navigator.pop(context),
                  style: OutlinedButton.styleFrom(
                    side: const BorderSide(color: Color(0xFF5B4FCF)),
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(30)),
                  ),
                  child: const Text('Generate Another',
                      style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w600,
                          color: Color(0xFF5B4FCF))),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  void _showRatingSheet(BuildContext context) {
    int selectedStars = 0;
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (_) => StatefulBuilder(
        builder: (context, setModalState) => Container(
          padding: const EdgeInsets.all(24),
          decoration: const BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.vertical(top: Radius.circular(24))),
          child: Column(mainAxisSize: MainAxisSize.min, children: [
            Container(
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                    color: const Color(0xFFE0E0E0),
                    borderRadius: BorderRadius.circular(2))),
            const SizedBox(height: 20),
            const Text('Rate this video',
                style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF1A1A2E))),
            const SizedBox(height: 20),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: List.generate(
                  5,
                  (i) => GestureDetector(
                        onTap: () => setModalState(() => selectedStars = i + 1),
                        child: Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 6),
                          child: Icon(
                            i < selectedStars
                                ? Icons.star_rounded
                                : Icons.star_outline_rounded,
                            size: 40,
                            color: const Color(0xFFFFB800),
                          ),
                        ),
                      )),
            ),
            const SizedBox(height: 24),
            CustomButton(
                text: 'Submit',
                onPressed: () async {
                  // Ici tu peux appeler un endpoint de notation si disponible
                  // Ex: await ApiService.rateGeneration(generationId, selectedStars);
                  if (context.mounted) Navigator.pop(context);
                }),
            const SizedBox(height: 12),
          ]),
        ),
      ),
    );
  }
}

class _ActionButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;
  const _ActionButton(
      {required this.icon, required this.label, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: GestureDetector(
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 12),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: const Color(0xFFF0EFF8)),
            boxShadow: [
              BoxShadow(
                  color: Colors.black.withOpacity(0.04),
                  blurRadius: 6,
                  offset: const Offset(0, 2))
            ],
          ),
          child: Column(children: [
            Icon(icon, size: 22, color: const Color(0xFF1A1A2E)),
            const SizedBox(height: 4),
            Text(label,
                style: const TextStyle(
                    fontSize: 12,
                    color: Color(0xFF1A1A2E),
                    fontWeight: FontWeight.w500)),
          ]),
        ),
      ),
    );
  }
}
