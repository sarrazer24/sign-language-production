import 'package:flutter/material.dart';
import '../../services/api_service.dart';
import '../../widgets/custom_text_field.dart';
import '../../widgets/custom_button.dart';
import 'sign_in_screen.dart';
import 'reset_password_screen.dart';

class ForgotPasswordScreen extends StatefulWidget {
  const ForgotPasswordScreen({super.key});

  @override
  State<ForgotPasswordScreen> createState() => _ForgotPasswordScreenState();
}

class _ForgotPasswordScreenState extends State<ForgotPasswordScreen> {
  final TextEditingController _emailController = TextEditingController();
  bool _isLoading = false;

  @override
  void dispose() {
    _emailController.dispose();
    super.dispose();
  }

  void _sendResetLink() async {
    final email = _emailController.text.trim();
    if (email.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please enter your email')),
      );
      return;
    }

    setState(() => _isLoading = true);

    try {
      final result = await ApiService.forgotPassword(email: email);
      if (!mounted) return;

      if (result['status'] == 200) {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (_) => ResetPasswordScreen(email: email),
          ),
        );
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(result['data']['error'] ?? 'Something went wrong'),
            backgroundColor: Colors.red,
          ),
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
              const SizedBox(height: 80),
              const Text('Forgot Password?',
                  style: TextStyle(fontSize: 30, fontWeight: FontWeight.bold,
                      color: Color(0xFF1A1A2E)),
                  textAlign: TextAlign.center),
              const SizedBox(height: 16),
              const Text("No worries! Enter your email and\nwe'll send you a reset link.",
                  style: TextStyle(fontSize: 15, color: Colors.grey, height: 1.5),
                  textAlign: TextAlign.center),
              const SizedBox(height: 48),
              CustomTextField(label: 'Email', hintText: 'Enter your email',
                  controller: _emailController, keyboardType: TextInputType.emailAddress),
              const SizedBox(height: 28),
              CustomButton(text: 'Send Reset Link', onPressed: _sendResetLink,
                  isLoading: _isLoading),
              const SizedBox(height: 280),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Text('Remember your password? ',
                      style: TextStyle(fontSize: 14, color: Colors.grey)),
                  GestureDetector(
                    onTap: () => Navigator.pushReplacement(context,
                        MaterialPageRoute(builder: (_) => const SignInScreen())),
                    child: const Text('Sign In',
                        style: TextStyle(fontSize: 14, color: Color(0xFF5B4FCF),
                            fontWeight: FontWeight.w600)),
                  ),
                ],
              ),
              const SizedBox(height: 32),
            ],
          ),
        ),
      ),
    );
  }
}