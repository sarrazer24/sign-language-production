import 'package:flutter/material.dart';
import 'auth/sign_in_screen.dart';
import '../services/api_service.dart';
import '../widgets/main_navigation.dart';

class SettingsScreen extends StatefulWidget {
  final String fullName;

  const SettingsScreen({super.key, required this.fullName});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  String email = '';

  @override
  void initState() {
    super.initState();
    _loadUser();
  }

  Future<void> _loadUser() async {
    final user = await ApiService.getUser();
    if (mounted) {
      setState(() {
        email = user?['email'] ?? '';
      });
    }
  }

  String get _firstLetter {
    final parts = widget.fullName.trim().split(' ');
    final firstName = parts.length > 1 ? parts[1] : parts[0];
    return firstName.isNotEmpty ? firstName[0].toUpperCase() : 'U';
  }

  static final List<Map<String, dynamic>> _sections = [
    {
      'title': 'Account',
      'items': [
        {'icon': Icons.person_outline, 'label': 'Profile'},
        {'icon': Icons.lock_outline, 'label': 'Privacy'},
        {'icon': Icons.notifications_outlined, 'label': 'Notifications'},
      ],
    },
    {
      'title': 'Preferences',
      'items': [
        {'icon': Icons.language_outlined, 'label': 'Language'},
        {'icon': Icons.dark_mode_outlined, 'label': 'Dark Mode'},
        {'icon': Icons.storage_outlined, 'label': 'Storage'},
      ],
    },
    {
      'title': 'Support',
      'items': [
        {'icon': Icons.help_outline, 'label': 'Help Center'},
        {'icon': Icons.star_outline, 'label': 'Rate the App'},
        {'icon': Icons.info_outline, 'label': 'About'},
      ],
    },
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF8F7FF),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const SizedBox(height: 24),

              // ── Title avec flèche retour ──
              Row(
                children: [
                  GestureDetector(
                    onTap: () => Navigator.pushAndRemoveUntil(
                      context,
                      MaterialPageRoute(builder: (_) => const MainNavigation()),
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
                  const Text(
                    'Settings',
                    style: TextStyle(
                        fontSize: 26,
                        fontWeight: FontWeight.bold,
                        color: Color(0xFF1A1A2E)),
                  ),
                ],
              ),

              const SizedBox(height: 24),

              // ── Profile card ──
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(18),
                  boxShadow: [
                    BoxShadow(
                        color: Colors.black.withValues(alpha: 0.04),
                        blurRadius: 8,
                        offset: const Offset(0, 2))
                  ],
                ),
                child: Row(
                  children: [
                    CircleAvatar(
                      radius: 28,
                      backgroundColor: const Color(0xFF5B4FCF),
                      child: Text(
                        _firstLetter,
                        style: const TextStyle(
                            color: Colors.white,
                            fontSize: 22,
                            fontWeight: FontWeight.bold),
                      ),
                    ),
                    const SizedBox(width: 14),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            widget.fullName,
                            style: const TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.w700,
                                color: Color(0xFF1A1A2E)),
                          ),
                          const SizedBox(height: 2),
                          Text(email,
                              style: const TextStyle(
                                  fontSize: 13, color: Colors.grey)),
                        ],
                      ),
                    ),
                    const Icon(Icons.chevron_right_rounded, color: Colors.grey),
                  ],
                ),
              ),

              const SizedBox(height: 24),

              // ── Sections ──
              ..._sections.map((section) {
                return Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(section['title'] as String,
                        style: const TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w700,
                            color: Color(0xFF9E9E9E),
                            letterSpacing: 0.8)),
                    const SizedBox(height: 10),
                    Container(
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(18),
                        boxShadow: [
                          BoxShadow(
                              color: Colors.black.withValues(alpha: 0.04),
                              blurRadius: 8,
                              offset: const Offset(0, 2))
                        ],
                      ),
                      child: Column(
                        children: List.generate(
                          (section['items'] as List).length,
                          (i) {
                            final item = (section['items'] as List)[i] as Map;
                            final isLast =
                                i == (section['items'] as List).length - 1;
                            return Column(
                              children: [
                                ListTile(
                                  leading: Container(
                                    padding: const EdgeInsets.all(8),
                                    decoration: BoxDecoration(
                                        color: const Color(0xFFF0EFF8),
                                        borderRadius:
                                            BorderRadius.circular(10)),
                                    child: Icon(item['icon'] as IconData,
                                        size: 18,
                                        color: const Color(0xFF5B4FCF)),
                                  ),
                                  title: Text(item['label'] as String,
                                      style: const TextStyle(
                                          fontSize: 14,
                                          fontWeight: FontWeight.w500,
                                          color: Color(0xFF1A1A2E))),
                                  trailing: const Icon(
                                      Icons.chevron_right_rounded,
                                      color: Colors.grey,
                                      size: 20),
                                  onTap: () {},
                                ),
                                if (!isLast)
                                  const Divider(
                                      height: 1,
                                      indent: 56,
                                      color: Color(0xFFF0EFF8)),
                              ],
                            );
                          },
                        ),
                      ),
                    ),
                    const SizedBox(height: 20),
                  ],
                );
              }),

              // ── Logout ──
              SizedBox(
                width: double.infinity,
                height: 52,
                child: OutlinedButton.icon(
                  onPressed: () async {
                    await ApiService.logout();
                    if (context.mounted) {
                      Navigator.pushAndRemoveUntil(
                        context,
                        MaterialPageRoute(builder: (_) => const SignInScreen()),
                        (route) => false,
                      );
                    }
                  },
                  style: OutlinedButton.styleFrom(
                    side: const BorderSide(color: Color(0xFFFFEBEE)),
                    backgroundColor: const Color(0xFFFFEBEE),
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(30)),
                  ),
                  icon: const Icon(Icons.logout_rounded,
                      color: Color(0xFFE53935), size: 18),
                  label: const Text('Log Out',
                      style: TextStyle(
                          fontSize: 15,
                          fontWeight: FontWeight.w600,
                          color: Color(0xFFE53935))),
                ),
              ),

              const SizedBox(height: 32),
            ],
          ),
        ),
      ),
    );
  }
}
