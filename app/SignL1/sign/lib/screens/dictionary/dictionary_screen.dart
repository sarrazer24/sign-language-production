import 'package:flutter/material.dart';
import '../../../widgets/sign_card.dart';

class DictionaryScreen extends StatefulWidget {
  const DictionaryScreen({super.key});

  @override
  State<DictionaryScreen> createState() => _DictionaryScreenState();
}

class _DictionaryScreenState extends State<DictionaryScreen> {
  final TextEditingController _searchController = TextEditingController();
  int _selectedTab = 0;
  String _searchQuery = '';

  // ── Dynamic tabs ──
  static const List<String> tabs = ['Alphabet', 'Numbers', 'Phrases'];

  // ── Dynamic alphabet data ──
  static final List<Map<String, String>> alphabetSigns = List.generate(
    26,
    (i) => {
      'letter': String.fromCharCode(65 + i),
      'image': 'assets/images/asl_${String.fromCharCode(97 + i)}.png',
    },
  );

  // ── Dynamic numbers data ──
  static final List<Map<String, String>> numberSigns = List.generate(
    10,
    (i) => {
      'letter': '$i',
      'image': 'assets/images/asl_num_$i.png',
    },
  );

  // ── Dynamic phrases data ──
  static final List<Map<String, String>> phraseSigns = [
    {'letter': 'Hello', 'image': 'assets/images/asl_hello.png'},
    {'letter': 'Thank You', 'image': 'assets/images/asl_thankyou.png'},
    {'letter': 'Please', 'image': 'assets/images/asl_please.png'},
    {'letter': 'Sorry', 'image': 'assets/images/asl_sorry.png'},
    {'letter': 'Yes', 'image': 'assets/images/asl_yes.png'},
    {'letter': 'No', 'image': 'assets/images/asl_no.png'},
  ];

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
                  // ── Title ──
                  const Text(
                    'Dictionary',
                    style: TextStyle(
                      fontSize: 26,
                      fontWeight: FontWeight.bold,
                      color: Color(0xFF1A1A2E),
                    ),
                  ),
                  const SizedBox(height: 4),
                  const Text(
                    'Learn ASL signs for everyday communication',
                    style: TextStyle(fontSize: 13, color: Colors.grey),
                  ),
                  const SizedBox(height: 16),

                  // ── Search bar ──
                  TextField(
                    controller: _searchController,
                    onChanged: (val) =>
                        setState(() => _searchQuery = val),
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
                        borderSide: BorderSide.none,
                      ),
                      enabledBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(30),
                        borderSide: BorderSide.none,
                      ),
                    ),
                  ),

                  const SizedBox(height: 16),

                  // ── Tabs ──
                  Row(
                    children: List.generate(tabs.length, (index) {
                      final isSelected = index == _selectedTab;
                      return GestureDetector(
                        onTap: () =>
                            setState(() => _selectedTab = index),
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
                          child: Text(
                            tabs[index],
                            style: TextStyle(
                              fontSize: 14,
                              fontWeight: FontWeight.w600,
                              color: isSelected
                                  ? Colors.white
                                  : const Color(0xFF1A1A2E),
                            ),
                          ),
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
                  ? const Center(
                      child: Text('No signs found',
                          style: TextStyle(color: Colors.grey)),
                    )
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
                        return SignCard(
                          letter: item['letter']!,
                          imagePath: item['image'],
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
