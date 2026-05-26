import 'package:flutter/material.dart';
import 'package:sign/screens/activity_screen.dart';
import 'package:sign/screens/ai_studio_screen.dart';
import 'package:sign/screens/dictionary/dictionary_screen.dart';
import 'package:sign/screens/home/home_screen.dart';
import 'package:sign/screens/settings_screen.dart';
import 'package:sign/widgets/bottom_nav_bar.dart';


class MainNavigation extends StatefulWidget {
  const MainNavigation({super.key});

  @override
  State<MainNavigation> createState() => _MainNavigationState();
}

class _MainNavigationState extends State<MainNavigation> {
  // ── currentIndex map ──
  // 0 = Home
  // 1 = Dictionary
  // 2 = AI Studio (centre)
  // 3 = Activity
  // 4 = Settings
  int _currentIndex = 0;

  // ── 5 screens, index 1:1 avec les tabs ──
  static const List<Widget> _screens = [
    HomeScreen(),        // index 0
    DictionaryScreen(),  // index 1
    AiStudioScreen(),    // index 2  ← centre
    ActivityScreen(),    // index 3
    SettingsScreen(),    // index 4
  ];

  void _onTabTapped(int index) {
    setState(() => _currentIndex = index);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // IndexedStack garde tous les screens en mémoire
      // → pas de rebuild quand on change d'onglet
      body: IndexedStack(
        index: _currentIndex,
        children: _screens,
      ),
      bottomNavigationBar: BottomNavBar(
        currentIndex: _currentIndex,
        onTap: _onTabTapped,
      ),
    );
  }
}
