import 'package:flutter/material.dart';
import 'package:sign/screens/home/home_screen.dart';
import '../../widgets/activity_card.dart';
import '../../services/api_service.dart';

class ActivityScreen extends StatefulWidget {
  const ActivityScreen({super.key});

  @override
  State<ActivityScreen> createState() => _ActivityScreenState();
}

class _ActivityScreenState extends State<ActivityScreen>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;

  List<ActivityItem> _items = [];
  bool _isLoading = true;

  // ════════════════════════════════
  // GETTERS
  // ════════════════════════════════
  List<ActivityItem> get _notifications =>
      _items.where((i) => i.type == ActivityType.notification).toList();

  List<ActivityItem> get _history =>
      _items.where((i) => i.type == ActivityType.history).toList();

  List<ActivityItem> get _currentList =>
      _tabController.index == 0 ? _notifications : _history;

  bool get _isNotifTab => _tabController.index == 0;

  bool get _hasUnread => _items.any((i) => i.isUnread);

  // ════════════════════════════════
  // LOAD DATA — vrais appels API ✅
  // ════════════════════════════════
  Future<void> _loadData() async {
    setState(() => _isLoading = true);
    try {
      // ✅ Appel API notifications
      final notifRaw = await ApiService.getNotifications();

      // ✅ Appel API historique des générations
      final historyRaw = await ApiService.getMyGenerations();

      final notifications = notifRaw.map<ActivityItem>((n) {
        return ActivityItem(
          id: n['id']?.toString() ?? UniqueKey().toString(),
          title: n['title'] ?? 'Notification',
          description: n['message'] ?? '',
          time: _formatTime(n['created_at']),
          status: _parseStatus(n['type']),
          type: ActivityType.notification,
          isUnread: !(n['is_read'] ?? false),
        );
      }).toList();

      final history = historyRaw.map<ActivityItem>((g) {
        final failed = g['status'] == 'failed';
        return ActivityItem(
          id: g['id']?.toString() ?? UniqueKey().toString(),
          title: failed ? 'Generation Failed' : 'Video Generated',
          description: failed
              ? '"${g['original_text']}" — failed'
              : '"${g['original_text']}" — ${g['duration'] ?? ''}',
          time: _formatTime(g['created_at']),
          status: failed ? ActivityStatus.error : ActivityStatus.success,
          type: ActivityType.history,
          isUnread: false,
        );
      }).toList();

      setState(() {
        _items = [...notifications, ...history];
        _isLoading = false;
      });
    } catch (e) {
      setState(() => _isLoading = false);
    }
  }

  String _formatTime(String? isoDate) {
    if (isoDate == null) return '';
    final date = DateTime.tryParse(isoDate);
    if (date == null) return '';
    final diff = DateTime.now().difference(date);
    if (diff.inMinutes < 60) return '${diff.inMinutes}m ago';
    if (diff.inHours < 24) return '${diff.inHours}h ago';
    if (diff.inDays == 1) return 'Yesterday';
    return '${diff.inDays} days ago';
  }

  ActivityStatus _parseStatus(String? type) {
    switch (type) {
      case 'success':
        return ActivityStatus.success;
      case 'error':
        return ActivityStatus.error;
      default:
        return ActivityStatus.info;
    }
  }

  // ════════════════════════════════
  // ACTIONS
  // ════════════════════════════════
  void _markAsRead(String id) {
    setState(() {
      final index = _items.indexWhere((i) => i.id == id);
      if (index != -1) _items[index].isUnread = false;
    });
  }

  void _markAllAsRead() {
    setState(() {
      for (final item in _items) {
        item.isUnread = false;
      }
    });
  }

  void _deleteItem(String id) {
    setState(() => _items.removeWhere((i) => i.id == id));
  }

  void _clearNotifications() async {
    // ✅ Appel API pour vider les notifications côté backend
    await ApiService.clearNotifications();
    setState(
        () => _items.removeWhere((i) => i.type == ActivityType.notification));
  }

  void _clearHistory() {
    setState(() => _items.removeWhere((i) => i.type == ActivityType.history));
  }

  // ════════════════════════════════
  // LIFECYCLE
  // ════════════════════════════════
  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 2, vsync: this);
    _tabController.addListener(() => setState(() {}));
    _loadData();
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  // ════════════════════════════════
  // BUILD
  // ════════════════════════════════
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF8F7FF),
      body: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // ── Header ──
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 24, 20, 0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Activity',
                    style: TextStyle(
                      fontSize: 26,
                      fontWeight: FontWeight.bold,
                      color: Color(0xFF1A1A2E),
                    ),
                  ),
                  const SizedBox(height: 4),
                  const Text(
                    'Notifications and generation history',
                    style: TextStyle(fontSize: 13, color: Colors.grey),
                  ),
                  const SizedBox(height: 20),

                  // ── Tab switcher ──
                  Container(
                    padding: const EdgeInsets.all(4),
                    decoration: BoxDecoration(
                      color: const Color(0xFFF0EFF8),
                      borderRadius: BorderRadius.circular(30),
                    ),
                    child: Row(
                      children: [
                        _buildTab('Notifications', 0),
                        _buildTab('History', 1),
                      ],
                    ),
                  ),

                  // ── Action row ──
                  if (!_isLoading && _isNotifTab && _currentList.isNotEmpty)
                    Padding(
                      padding: const EdgeInsets.only(top: 12),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          if (_hasUnread)
                            GestureDetector(
                              onTap: _markAllAsRead,
                              child: const Text(
                                'Mark all read',
                                style: TextStyle(
                                  fontSize: 13,
                                  color: Color(0xFF5B4FCF),
                                  fontWeight: FontWeight.w500,
                                ),
                              ),
                            )
                          else
                            const SizedBox(),
                          GestureDetector(
                            onTap: () => _confirmDialog(
                              context,
                              title: 'Clear all notifications?',
                              content: 'This action cannot be undone.',
                              onConfirm: _clearNotifications,
                            ),
                            child: const Text(
                              'Clear all',
                              style: TextStyle(
                                fontSize: 13,
                                color: Color(0xFF5B4FCF),
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ),
                        ],
                      ),
                    )
                  else if (!_isLoading &&
                      !_isNotifTab &&
                      _currentList.isNotEmpty)
                    Padding(
                      padding: const EdgeInsets.only(top: 12),
                      child: Align(
                        alignment: Alignment.centerRight,
                        child: GestureDetector(
                          onTap: () => _confirmDialog(
                            context,
                            title: 'Clear history?',
                            content: 'All records will be removed.',
                            onConfirm: _clearHistory,
                          ),
                          child: const Text(
                            'Clear history',
                            style: TextStyle(
                              fontSize: 13,
                              color: Color(0xFF5B4FCF),
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                      ),
                    )
                  else
                    const SizedBox(height: 12),
                ],
              ),
            ),

            // ── List / Loading / Empty state ──
            Expanded(
              child: _isLoading
                  ? const Center(
                      child: CircularProgressIndicator(
                        color: Color(0xFF5B4FCF),
                      ),
                    )
                  : RefreshIndicator(
                      color: const Color(0xFF5B4FCF),
                      onRefresh: _loadData, // ✅ Pull to refresh
                      child: AnimatedSwitcher(
                        duration: const Duration(milliseconds: 300),
                        child: _currentList.isEmpty
                            ? _EmptyState(isNotifTab: _isNotifTab)
                            : ListView.builder(
                                key: ValueKey(
                                    '${_tabController.index}_${_currentList.length}'),
                                padding:
                                    const EdgeInsets.fromLTRB(20, 8, 20, 20),
                                itemCount: _currentList.length,
                                itemBuilder: (context, index) {
                                  final item = _currentList[index];
                                  return ActivityCard(
                                    item: item,
                                    onTap: () {
                                      _markAsRead(item.id);
                                      Navigator.of(context).push(
                                        MaterialPageRoute(
                                          builder: (_) => const HomeScreen(),
                                        ),
                                      );
                                    },
                                    onDismiss: () => _deleteItem(item.id),
                                  );
                                },
                              ),
                      ),
                    ),
            ),
          ],
        ),
      ),
    );
  }

  // ── Tab pill button ──
  Widget _buildTab(String label, int index) {
    final isSelected = _tabController.index == index;
    return Expanded(
      child: GestureDetector(
        onTap: () {
          _tabController.animateTo(index);
          setState(() {});
        },
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 250),
          padding: const EdgeInsets.symmetric(vertical: 10),
          decoration: BoxDecoration(
            color: isSelected ? const Color(0xFF5B4FCF) : Colors.transparent,
            borderRadius: BorderRadius.circular(26),
          ),
          child: Text(
            label,
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w600,
              color: isSelected ? Colors.white : const Color(0xFF9E9E9E),
            ),
          ),
        ),
      ),
    );
  }

  // ── Confirm dialog ──
  void _confirmDialog(
    BuildContext context, {
    required String title,
    required String content,
    required VoidCallback onConfirm,
  }) {
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: Text(title),
        content: Text(content),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel', style: TextStyle(color: Colors.grey)),
          ),
          TextButton(
            onPressed: () {
              onConfirm();
              Navigator.pop(context);
            },
            child: const Text(
              'Confirm',
              style: TextStyle(
                color: Color(0xFFE53935),
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ════════════════════════════════
// EMPTY STATE WIDGET
// ════════════════════════════════
class _EmptyState extends StatelessWidget {
  final bool isNotifTab;
  const _EmptyState({required this.isNotifTab});

  @override
  Widget build(BuildContext context) {
    return Center(
      key: const ValueKey('empty'),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          SizedBox(
            width: 90,
            height: 90,
            child: Stack(
              alignment: Alignment.center,
              children: [
                Container(
                  width: 90,
                  height: 90,
                  decoration: const BoxDecoration(
                    color: Color(0xFFF0EFF8),
                    shape: BoxShape.circle,
                  ),
                ),
                Icon(
                  isNotifTab
                      ? Icons.notifications_outlined
                      : Icons.history_rounded,
                  size: 44,
                  color: const Color(0xFFBBB8E8),
                ),
                if (isNotifTab)
                  Transform.rotate(
                    angle: 0.65,
                    child: Container(
                      width: 2.5,
                      height: 56,
                      decoration: BoxDecoration(
                        color: const Color(0xFFBBB8E8),
                        borderRadius: BorderRadius.circular(2),
                      ),
                    ),
                  ),
              ],
            ),
          ),
          const SizedBox(height: 20),
          Text(
            isNotifTab ? 'No notifications' : 'No history yet',
            style: const TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w600,
              color: Color(0xFF1A1A2E),
            ),
          ),
          const SizedBox(height: 8),
          Text(
            isNotifTab
                ? "You're all caught up!\nNew notifications will appear here."
                : "Your generated videos will\nappear here.",
            textAlign: TextAlign.center,
            style:
                const TextStyle(fontSize: 13, color: Colors.grey, height: 1.5),
          ),
        ],
      ),
    );
  }
}
