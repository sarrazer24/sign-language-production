import 'package:flutter/material.dart';

// ════════════════════════════════════════
// ENUMS  — importés dans activity_screen
// ════════════════════════════════════════
enum ActivityStatus { success, info, error }

enum ActivityType { notification, history }

// ════════════════════════════════════════
// MODEL ActivityItem
// ════════════════════════════════════════
class ActivityItem {
  final String id;
  final String title;
  final String description;
  final String time;
  final ActivityStatus status;
  final ActivityType type;
  bool isUnread;

  ActivityItem({
    required this.id,
    required this.title,
    required this.description,
    required this.time,
    required this.status,
    required this.type,
    this.isUnread = false,
  });
}

// ════════════════════════════════════════
// WIDGET ActivityCard
// ════════════════════════════════════════
class ActivityCard extends StatelessWidget {
  final ActivityItem item;
  final VoidCallback? onTap;
  final VoidCallback? onDismiss;

  const ActivityCard({
    super.key,
    required this.item,
    this.onTap,
    this.onDismiss,
  });

  @override
  Widget build(BuildContext context) {
    return Dismissible(
      key: Key(item.id),
      direction: DismissDirection.endToStart,
      onDismissed: (_) => onDismiss?.call(),
      background: Container(
        alignment: Alignment.centerRight,
        padding: const EdgeInsets.only(right: 20),
        margin: const EdgeInsets.only(bottom: 12),
        decoration: BoxDecoration(
          color: const Color(0xFFFFEBEE),
          borderRadius: BorderRadius.circular(16),
        ),
        child: const Icon(Icons.delete_outline,
            color: Color(0xFFE53935), size: 24),
      ),
      child: GestureDetector(
        onTap: onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 300),
          margin: const EdgeInsets.only(bottom: 12),
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: item.isUnread ? const Color(0xFFFAF9FF) : Colors.white,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: item.isUnread
                  ? const Color(0xFFD6D1F5)
                  : const Color(0xFFF0EFF8),
            ),
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
              _StatusIcon(status: item.status),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      item.title,
                      style: TextStyle(
                        fontSize: 14,
                        fontWeight: item.isUnread
                            ? FontWeight.w700
                            : FontWeight.w600,
                        color: const Color(0xFF1A1A2E),
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      item.description,
                      style: const TextStyle(
                          fontSize: 13, color: Colors.grey, height: 1.4),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      item.time,
                      style: const TextStyle(
                          fontSize: 12, color: Color(0xFFAAAAAA)),
                    ),
                  ],
                ),
              ),
              if (item.isUnread)
                Container(
                  width: 8,
                  height: 8,
                  margin: const EdgeInsets.only(top: 4, left: 8),
                  decoration: const BoxDecoration(
                    shape: BoxShape.circle,
                    color: Color(0xFF5B4FCF),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}

// ── Status icon ──
class _StatusIcon extends StatelessWidget {
  final ActivityStatus status;
  const _StatusIcon({required this.status});

  @override
  Widget build(BuildContext context) {
    switch (status) {
      case ActivityStatus.success:
        return const CircleAvatar(
          radius: 16,
          backgroundColor: Color(0xFFE8F5E9),
          child: Icon(Icons.check_circle_outline,
              color: Color(0xFF4CAF50), size: 18),
        );
      case ActivityStatus.info:
        return const CircleAvatar(
          radius: 16,
          backgroundColor: Color(0xFFE8EAF6),
          child: Icon(Icons.info_outline, color: Color(0xFF5B4FCF), size: 18),
        );
      case ActivityStatus.error:
        return const CircleAvatar(
          radius: 16,
          backgroundColor: Color(0xFFFFEBEE),
          child: Icon(Icons.cancel_outlined,
              color: Color(0xFFE53935), size: 18),
        );
    }
  }
}
