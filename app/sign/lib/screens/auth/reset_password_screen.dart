import 'package:flutter/material.dart';
import '../../services/api_service.dart';
import '../../widgets/custom_text_field.dart';
import '../../widgets/custom_button.dart';
import 'sign_in_screen.dart';

class ResetPasswordScreen extends StatefulWidget {
  final String email;
  const ResetPasswordScreen({super.key, required this.email});

  @override
  State<ResetPasswordScreen> createState() => _ResetPasswordScreenState();
}

class _ResetPasswordScreenState extends State<ResetPasswordScreen> {
  final TextEditingController _passwordController = TextEditingController();
  final TextEditingController _confirmController = TextEditingController();
  bool _isLoading = false;

  @override
  void dispose() {
    _passwordController.dispose();
    _confirmController.dispose();
    super.dispose();
  }

  void _resetPassword() async {
    if (_passwordController.text != _confirmController.text) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Passwords do not match'),
            backgroundColor: Colors.red),
      );
      return;
    }
    if (_passwordController.text.length < 6) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Password must be at least 6 characters'),
            backgroundColor: Colors.red),
      );
      return;
    }

    setState(() => _isLoading = true);

    try {
      final result = await ApiService.resetPassword(
        email: widget.email,
        newPassword: _passwordController.text,
      );
      if (!mounted) return;

      if (result['status'] == 200) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Password reset successfully!'),
              backgroundColor: Colors.green),
        );
        Navigator.pushAndRemoveUntil(
          context,
          MaterialPageRoute(builder: (_) => const SignInScreen()),
          (route) => false,
        );
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text(result['data']['error'] ?? 'Error'),
              backgroundColor: Colors.red),
        );
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: $e'), backgroundColor: Colors.red),
      );
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 28.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              const SizedBox(height: 60),
              GestureDetector(
                onTap: () => Navigator.pop(context),
                child: Container(
                  width: 36, height: 36,
                  decoration: BoxDecoration(color: const Color(0xFFF0EFF8),
                      borderRadius: BorderRadius.circular(10)),
                  child: const Icon(Icons.arrow_back_ios_new_rounded,
                      size: 16, color: Color(0xFF5B4FCF)),
                ),
              ),
              const SizedBox(height: 32),
              const Text('New Password',
                  style: TextStyle(fontSize: 30, fontWeight: FontWeight.bold,
                      color: Color(0xFF1A1A2E)),
                  textAlign: TextAlign.center),
              const SizedBox(height: 16),
              Text('Enter your new password for\n${widget.email}',
                  style: const TextStyle(fontSize: 15, color: Colors.grey, height: 1.5),
                  textAlign: TextAlign.center),
              const SizedBox(height: 48),
              CustomTextField(label: 'New Password', hintText: 'Enter new password',
                  controller: _passwordController, obscureText: true),
              const SizedBox(height: 20),
              CustomTextField(label: 'Confirm Password', hintText: 'Confirm new password',
                  controller: _confirmController, obscureText: true),
              const SizedBox(height: 28),
              CustomButton(text: 'Reset Password', onPressed: _resetPassword,
                  isLoading: _isLoading),
              const SizedBox(height: 32),
            ],
          ),
        ),
      ),
    );
  }
}