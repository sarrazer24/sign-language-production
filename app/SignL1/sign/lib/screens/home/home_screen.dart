import 'package:flutter/material.dart';
import '../../../widgets/recent_activity_card.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  // ── Dynamic data ──
  static const String userName = 'Adam';
  static const String subtitle = 'Ready to communicate today?';

  static final List<Map<String, String>> recentActivities = [
    {'text': 'Hello, how are you today?', 'time': '50m ago'},
    {'text': 'Good morning everyone!', 'time': '2h ago'},
    {'text': 'Thank you very much', 'time': '5h ago'},
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

              // ── Header ──
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Hello, $userName',
                        style: const TextStyle(
                          fontSize: 26,
                          fontWeight: FontWeight.bold,
                          color: Color(0xFF1A1A2E),
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        subtitle,
                        style: const TextStyle(
                          fontSize: 14,
                          color: Colors.grey,
                        ),
                      ),
                    ],
                  ),
                  CircleAvatar(
                    radius: 22,
                    backgroundColor: const Color(0xFF5B4FCF),
                    child: Text(
                      userName[0],
                      style: const TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                        fontSize: 18,
                      ),
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
                      offset: const Offset(0, 2),
                    ),
                  ],
                ),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [

                    Container(
                      padding: const EdgeInsets.all(2),
                     
                      child: const Icon(
                        Icons.back_hand_outlined,
                        color: Color(0xFF5B4FCF),
                        size: 20,
                      ),
                    ),
                    const SizedBox(width: 12),
                    const Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'AI Sign Language',
                            style: TextStyle(
                              fontSize: 14,
                              fontWeight: FontWeight.w600,
                              color: Color(0xFF1A1A2E),
                            ),
                          ),
                          SizedBox(height: 4),
                          Text(
                            'Convert text or voice messages into realistic sign language videos. Type or speak, and our AI pipeline handles the rest.',
                            style: TextStyle(
                              fontSize: 12,
                              color: Colors.grey,
                              height: 1.4,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 16),

              // ── Start Generating button ──
              Container(
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
                      offset: const Offset(0, 6),
                    ),
                  ],
                ),
                child: Row(
                  
                  children: [
                    const Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Icon(Icons.auto_awesome,
                                  color: Colors.white70, size: 16),
                              SizedBox(width: 12),
                              Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  
                              Text(
                                'Start Generating',
                                style: TextStyle(
                                  fontSize: 18,
                                  fontWeight: FontWeight.bold,
                                  color: Colors.white,
                                ),
                              ),
                              SizedBox(height: 4),
                          Text(
                            'Create your video',
                            style: TextStyle(
                              fontSize: 13,
                              color: Colors.white70,
                            ),
                          ),
                                ],
                              )
                             
                            ],
                          ),
                          SizedBox(height: 4),
                          
                        ],
                      ),
                    ),
                    Container(   //////////////////////////////////////////////////////////////////
                      padding: const EdgeInsets.all(10),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        shape: BoxShape.circle,
                      ),
                      child: const Icon(Icons.arrow_forward_rounded,
                          color: Color(0xFF5B4FCF), size: 20),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 28),

              // ── Recent Activity header ──
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text(
                    'RECENT ACTIVITY',
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w700,
                      color: Color(0xFF9E9E9E),
                      letterSpacing: 1.0,
                    ),
                  ),
                  GestureDetector(
                    onTap: () {}, //TODO: Navigate to full activity screen
                    child: const Text(
                      'View all',
                      style: TextStyle(
                        fontSize: 13,
                        color: Color(0xFF5B4FCF),
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                ],
              ),

              const SizedBox(height: 12),

              // ── Recent activity list ── 
              ...recentActivities.map(
                (item) => Padding(
                  padding: const EdgeInsets.only(bottom: 10),
                  child: RecentActivityCard(
                    text: item['text']!,
                    time: item['time']!,
                  ),
                ),
              ),

              const SizedBox(height: 16),
            ],
          ),
        ),
      ),
    );
  }
}
