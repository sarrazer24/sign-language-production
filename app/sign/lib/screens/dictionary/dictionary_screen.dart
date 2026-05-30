import 'package:flutter/material.dart';
import '../../../widgets/main_navigation.dart';

class DictionaryScreen extends StatefulWidget {
  const DictionaryScreen({super.key});

  @override
  State<DictionaryScreen> createState() => _DictionaryScreenState();
}

class _DictionaryScreenState extends State<DictionaryScreen> {
  final TextEditingController _searchController = TextEditingController();
  int _selectedTab = 0;
  String _searchQuery = '';

  static const List<String> tabs = ['Alphabet', 'Numbers', 'Phrases'];

  static final List<Map<String, String>> alphabetSigns = List.generate(
    26,
    (i) => {
      'letter': String.fromCharCode(65 + i),
      'image': 'assets/images/asl_${String.fromCharCode(97 + i)}.png',
    },
  );

  static final List<Map<String, String>> numberSigns = List.generate(
    10,
    (i) => {
      'letter': '$i',
      'image': 'assets/images/asl_num_$i.png',
    },
  );

  static final List<Map<String, String>> phraseSigns = [];

  List<Map<String, String>> get _currentData {
    List<Map<String, String>> data;
    switch (_selectedTab) {
      case 0:
        data = alphabetSigns;
        break;
      case 1:
        data = numberSigns;
        break;
      default:
        data = phraseSigns;
    }
    if (_searchQuery.isEmpty) return data;
    return data
        .where((item) =>
            item['letter']!.toLowerCase().contains(_searchQuery.toLowerCase()))
        .toList();
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
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
              padding: const EdgeInsets.fromLTRB(20, 24, 20, 0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // ── Title avec flèche retour ──
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
                      const Text('Dictionary',
                          style: TextStyle(
                              fontSize: 26,
                              fontWeight: FontWeight.bold,
                              color: Color(0xFF1A1A2E))),
                    ],
                  ),
                  const SizedBox(height: 4),
                  const Padding(
                    padding: EdgeInsets.only(left: 48),
                    child: Text('Learn ASL signs for everyday communication',
                        style: TextStyle(fontSize: 13, color: Colors.grey)),
                  ),
                  const SizedBox(height: 16),

                  // ── Search bar ──
                  TextField(
                    controller: _searchController,
                    onChanged: (val) => setState(() => _searchQuery = val),
                    decoration: InputDecoration(
                      hintText: 'Search signs...',
                      hintStyle: const TextStyle(
                          color: Color(0xFFAAAAAA), fontSize: 14),
                      prefixIcon: const Icon(Icons.search,
                          color: Color(0xFFAAAAAA), size: 20),
                      filled: true,
                      fillColor: Colors.white,
                      contentPadding: const EdgeInsets.symmetric(
                          horizontal: 16, vertical: 14),
                      border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(30),
                          borderSide: BorderSide.none),
                      enabledBorder: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(30),
                          borderSide: BorderSide.none),
                    ),
                  ),
                  const SizedBox(height: 16),

                  // ── Tabs ──
                  Row(
                    children: List.generate(tabs.length, (index) {
                      final isSelected = index == _selectedTab;
                      return GestureDetector(
                        onTap: () => setState(() => _selectedTab = index),
                        child: AnimatedContainer(
                          duration: const Duration(milliseconds: 250),
                          margin: const EdgeInsets.only(right: 10),
                          padding: const EdgeInsets.symmetric(
                              horizontal: 20, vertical: 10),
                          decoration: BoxDecoration(
                            color: isSelected
                                ? const Color(0xFF5B4FCF)
                                : Colors.white,
                            borderRadius: BorderRadius.circular(30),
                          ),
                          child: Text(tabs[index],
                              style: TextStyle(
                                  fontSize: 14,
                                  fontWeight: FontWeight.w600,
                                  color: isSelected
                                      ? Colors.white
                                      : const Color(0xFF1A1A2E))),
                        ),
                      );
                    }),
                  ),
                  const SizedBox(height: 16),
                ],
              ),
            ),

            // ── Grid ──
            Expanded(
              child: _currentData.isEmpty
                  ? const SizedBox()
                  : GridView.builder(
                      padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
                      gridDelegate:
                          const SliverGridDelegateWithFixedCrossAxisCount(
                        crossAxisCount: 3,
                        crossAxisSpacing: 12,
                        mainAxisSpacing: 12,
                        childAspectRatio: 0.85,
                      ),
                      itemCount: _currentData.length,
                      itemBuilder: (context, index) {
                        final item = _currentData[index];
                        return Container(
                          decoration: BoxDecoration(
                            color: Colors.white,
                            borderRadius: BorderRadius.circular(16),
                            boxShadow: [
                              BoxShadow(
                                  color: Colors.black.withOpacity(0.05),
                                  blurRadius: 8,
                                  offset: const Offset(0, 2))
                            ],
                          ),
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Expanded(
                                child: ClipRRect(
                                  borderRadius: const BorderRadius.vertical(
                                      top: Radius.circular(16)),
                                  child: Image.asset(
                                    item['image']!,
                                    fit: BoxFit.cover,
                                    width: double.infinity,
                                    errorBuilder: (_, __, ___) => const Icon(
                                        Icons.back_hand_outlined,
                                        color: Color(0xFF5B4FCF),
                                        size: 40),
                                  ),
                                ),
                              ),
                              Padding(
                                padding:
                                    const EdgeInsets.symmetric(vertical: 8),
                                child: Text(item['letter']!,
                                    style: const TextStyle(
                                        fontSize: 14,
                                        fontWeight: FontWeight.w600,
                                        color: Color(0xFF1A1A2E))),
                              ),
                            ],
                          ),
                        );
                      },
                    ),
            ),
          ],
        ),
      ),
    );
  }
}
