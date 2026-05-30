import 'package:flutter/material.dart';
import 'package:sign/screens/settings_screen.dart';
import '../../../widgets/recent_activity_card.dart';
import '../../../services/api_service.dart';
import '../sign_to_text_screen.dart';
import '../text_to_sign_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  String userName = '';
  String fullName = '';
  List<dynamic> recentActivities = [];
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _loadData() async {
    final user = await ApiService.getUser();
    final activities = await ApiService.getActivity();
    if (mounted) {
      setState(() {
        final name = user?['full_name'] ?? 'User';
        fullName = name;
        final parts = name.trim().split(' ');
        final firstName = parts.length > 1 ? parts[1] : parts[0];
        userName =
            firstName[0].toUpperCase() + firstName.substring(1).toLowerCase();
        recentActivities = activities;
        _loading = false;
      });
    }
  }

  String get _firstLetter =>
      userName.isNotEmpty ? userName[0].toUpperCase() : 'U';

  void _showChoiceDialog(BuildContext context) {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (_) => Container(
        padding: const EdgeInsets.all(24),
        decoration: const BoxDecoration(
          color: Color(0xFFF8F7FF),
          borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Choose Mode',
                style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF1A1A2E))),
            const SizedBox(height: 6),
            const Text('What would you like to do?',
                style: TextStyle(fontSize: 13, color: Colors.grey)),
            const SizedBox(height: 24),

            // ── Text to Sign ──
            GestureDetector(
              onTap: () {
                Navigator.pop(context);
                Navigator.push(
                    context,
                    MaterialPageRoute(
                        builder: (_) => const TextToSignScreen()));
              },
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.all(18),
                decoration: BoxDecoration(
                  gradient: const LinearGradient(
                    colors: [Color(0xFF7B6EF6), Color(0xFF5B4FCF)],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                  borderRadius: BorderRadius.circular(16),
                  boxShadow: [
                    BoxShadow(
                        color: const Color(0xFF5B4FCF).withOpacity(0.3),
                        blurRadius: 12,
                        offset: const Offset(0, 4))
                  ],
                ),
                child: const Row(
                  children: [
                    Icon(Icons.text_fields_rounded,
                        color: Colors.white, size: 28),
                    SizedBox(width: 16),
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('Text to Sign',
                            style: TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.bold,
                                color: Colors.white)),
                        SizedBox(height: 2),
                        Text('Convert text into sign language video',
                            style:
                                TextStyle(fontSize: 12, color: Colors.white70)),
                      ],
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 14),

            // ── Sign to Text ──
            GestureDetector(
              onTap: () {
                Navigator.pop(context);
                Navigator.push(
                    context,
                    MaterialPageRoute(
                        builder: (_) => const SignToTextScreen()));
              },
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.all(18),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(16),
                  border:
                      Border.all(color: const Color(0xFF5B4FCF), width: 1.5),
                  boxShadow: [
                    BoxShadow(
                        color: Colors.black.withOpacity(0.04),
                        blurRadius: 8,
                        offset: const Offset(0, 2))
                  ],
                ),
                child: const Row(
                  children: [
                    Icon(Icons.sign_language_rounded,
                        color: Color(0xFF5B4FCF), size: 28),
                    SizedBox(width: 16),
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('Sign to Text',
                            style: TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.bold,
                                color: Color(0xFF1A1A2E))),
                        SizedBox(height: 2),
                        Text('Translate sign language to text',
                            style:
                                TextStyle(fontSize: 12, color: Colors.grey)),
                      ],
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF8F7FF),
      body: SafeArea(
        child: _loading
            ? const Center(
                child: CircularProgressIndicator(color: Color(0xFF5B4FCF)))
            : SingleChildScrollView(
                padding: const EdgeInsets.symmetric(horizontal: 20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const SizedBox(height: 24),

                    // ── Header ──
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text('Hello, $userName',
                                style: const TextStyle(
                                    fontSize: 26,
                                    fontWeight: FontWeight.bold,
                                    color: Color(0xFF1A1A2E))),
                            const SizedBox(height: 4),
                            const Text('Ready to communicate today?',
                                style: TextStyle(
                                    fontSize: 14, color: Colors.grey)),
                          ],
                        ),
                        GestureDetector(
                          onTap: () => Navigator.push(
                              context,
                              MaterialPageRoute(
                                  builder: (_) =>
                                      SettingsScreen(fullName: fullName))),
                          child: CircleAvatar(
                            radius: 22,
                            backgroundColor: const Color(0xFF5B4FCF),
                            child: Text(_firstLetter,
                                style: const TextStyle(
                                    color: Colors.white,
                                    fontWeight: FontWeight.bold,
                                    fontSize: 18)),
                          ),
                        ),
                      ],
                    ),

                    const SizedBox(height: 24),

                    // ── AI Sign Language info card ──
                    Container(
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(16),
                        border: Border.all(color: const Color(0xFF5B4FCF)),
                        boxShadow: [
                          BoxShadow(
                              color: Colors.black.withOpacity(0.04),
                              blurRadius: 8,
                              offset: const Offset(0, 2))
                        ],
                      ),
                      child: const Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Icon(Icons.back_hand_outlined,
                              color: Color(0xFF5B4FCF), size: 20),
                          SizedBox(width: 12),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text('AI Sign Language',
                                    style: TextStyle(
                                        fontSize: 14,
                                        fontWeight: FontWeight.w600,
                                        color: Color(0xFF1A1A2E))),
                                SizedBox(height: 4),
                                Text(
                                  'Convert text or voice messages into realistic sign language videos. Type or speak, and our AI pipeline handles the rest.',
                                  style: TextStyle(
                                      fontSize: 12,
                                      color: Colors.grey,
                                      height: 1.4),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),

                    const SizedBox(height: 16),

                    // ── Start Generating button ──
                    GestureDetector(
                      onTap: () => _showChoiceDialog(context),
                      child: Container(
                        width: double.infinity,
                        padding: const EdgeInsets.symmetric(
                            horizontal: 20, vertical: 18),
                        decoration: BoxDecoration(
                          gradient: const LinearGradient(
                            colors: [Color(0xFF7B6EF6), Color(0xFF5B4FCF)],
                            begin: Alignment.topLeft,
                            end: Alignment.bottomRight,
                          ),
                          borderRadius: BorderRadius.circular(18),
                          boxShadow: [
                            BoxShadow(
                                color: const Color(0xFF5B4FCF).withOpacity(0.35),
                                blurRadius: 16,
                                offset: const Offset(0, 6))
                          ],
                        ),
                        child: Row(
                          children: [
                            const Expanded(
                              child: Row(
                                children: [
                                  Icon(Icons.auto_awesome,
                                      color: Colors.white70, size: 16),
                                  SizedBox(width: 12),
                                  Column(
                                    crossAxisAlignment:
                                        CrossAxisAlignment.start,
                                    children: [
                                      Text('Start Generating',
                                          style: TextStyle(
                                              fontSize: 18,
                                              fontWeight: FontWeight.bold,
                                              color: Colors.white)),
                                      SizedBox(height: 4),
                                      Text('Create your video',
                                          style: TextStyle(
                                              fontSize: 13,
                                              color: Colors.white70)),
                                    ],
                                  ),
                                ],
                              ),
                            ),
                            Container(
                              padding: const EdgeInsets.all(10),
                              decoration: const BoxDecoration(
                                  color: Colors.white, shape: BoxShape.circle),
                              child: const Icon(Icons.arrow_forward_rounded,
                                  color: Color(0xFF5B4FCF), size: 20),
                            ),
                          ],
                        ),
                      ),
                    ),

                    const SizedBox(height: 28),

                    // ── Recent Activity ──
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        const Text('RECENT ACTIVITY',
                            style: TextStyle(
                                fontSize: 12,
                                fontWeight: FontWeight.w700,
                                color: Color(0xFF9E9E9E),
                                letterSpacing: 1.0)),
                        const Text('View all',
                            style: TextStyle(
                                fontSize: 13,
                                color: Color(0xFF5B4FCF),
                                fontWeight: FontWeight.w600)),
                      ],
                    ),

                    const SizedBox(height: 12),

                    recentActivities.isEmpty
                        ? const SizedBox()
                        : Column(
                            children: recentActivities
                                .map((item) => Padding(
                                      padding:
                                          const EdgeInsets.only(bottom: 10),
                                      child: RecentActivityCard(
                                        text: item['original_text'] ?? '',
                                        time: item['created_at'] ?? '',
                                      ),
                                    ))
                                .toList(),
                          ),

                    const SizedBox(height: 16),
                  ],
                ),
              ),
      ),
    );
  }
}