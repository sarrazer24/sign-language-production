import 'package:flutter/material.dart';
import 'package:sign/screens/activity_screen.dart';
import 'package:sign/screens/ai_studio_screen.dart';
import 'package:sign/screens/dictionary/dictionary_screen.dart';
import 'package:sign/screens/home/home_screen.dart';
import 'package:sign/screens/settings_screen.dart';
import 'package:sign/widgets/bottom_nav_bar.dart';
import 'package:sign/services/api_service.dart';

class MainNavigation extends StatefulWidget {
  const MainNavigation({super.key});

  @override
  State<MainNavigation> createState() => _MainNavigationState();
}

class _MainNavigationState extends State<MainNavigation> {
  int _currentIndex = 0;
  String _fullName = '';

  @override
  void initState() {
    super.initState();
    _loadUser();
  }

  Future<void> _loadUser() async {
    final user = await ApiService.getUser();
    if (mounted) {
      setState(() {
        _fullName = user?['full_name'] ?? '';
      });
    }
  }

  void _onTabTapped(int index) {
    setState(() => _currentIndex = index);
  }

  @override
  Widget build(BuildContext context) {
    final List<Widget> screens = [
      const HomeScreen(),
      const DictionaryScreen(),
      const AiStudioScreen(),
      const ActivityScreen(),
      SettingsScreen(fullName: _fullName),
    ];

    return Scaffold(
      body: IndexedStack(
        index: _currentIndex,
        children: screens,
      ),
      bottomNavigationBar: BottomNavBar(
        currentIndex: _currentIndex,
        onTap: _onTabTapped,
      ),
    );
  }
}