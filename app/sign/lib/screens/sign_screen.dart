import 'package:flutter/material.dart';
import '../services/sign_service.dart';
import '../widgets/sign_animator.dart';

class SignScreen extends StatefulWidget {
  const SignScreen({super.key});

  @override
  State<SignScreen> createState() => _SignScreenState();
}

class _SignScreenState extends State<SignScreen> {
  final _ctrl = TextEditingController();
  SignResult? _result;
  bool _loading = false;
  String? _error;

  Future<void> _generate() async {
    final text = _ctrl.text.trim();
    if (text.isEmpty) return;

    setState(() { _loading = true; _error = null; _result = null; });

    try {
      final result = await SignService.generate(text: text);
      setState(() { _result = result; });
    } catch (e) {
      setState(() { _error = e.toString(); });
    } finally {
      setState(() { _loading = false; });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Sign Language Generator')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            // Input row
            Row(children: [
              Expanded(
                child: TextField(
                  controller: _ctrl,
                  decoration: const InputDecoration(
                    hintText: 'Enter text (e.g. "hello how are you")',
                    border: OutlineInputBorder(),
                  ),
                  onSubmitted: (_) => _generate(),
                ),
              ),
              const SizedBox(width: 8),
              ElevatedButton(
                onPressed: _loading ? null : _generate,
                child: _loading
                    ? const SizedBox(
                        width: 20, height: 20,
                        child: CircularProgressIndicator(strokeWidth: 2))
                    : const Text('Generate'),
              ),
            ]),

            const SizedBox(height: 16),

            if (_error != null)
              Text(_error!, style: const TextStyle(color: Colors.red)),

            if (_result != null)
              Expanded(child: SingleChildScrollView(
                child: SignAnimator(result: _result!),
              )),
          ],
        ),
      ),
    );
  }
}