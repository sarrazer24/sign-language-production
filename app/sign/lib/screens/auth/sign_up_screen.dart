import 'package:flutter/material.dart';
import '../../widgets/custom_text_field.dart';
import '../../widgets/custom_button.dart';
import '../../services/api_service.dart';
import '../../widgets/main_navigation.dart';
import 'sign_in_screen.dart';

class SignUpScreen extends StatefulWidget {
  const SignUpScreen({super.key});

  @override
  State<SignUpScreen> createState() => _SignUpScreenState();
}

class _SignUpScreenState extends State<SignUpScreen> {
  final TextEditingController _nameController = TextEditingController();
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();
  final TextEditingController _confirmPasswordController = TextEditingController();
  bool _isLoading = false;
  String? _errorMessage;

  @override
  void dispose() {
    _nameController.dispose();
    _emailController.dispose();
    _passwordController.dispose();
    _confirmPasswordController.dispose();
    super.dispose();
  }

  void _signUp() async {
    if (_passwordController.text != _confirmPasswordController.text) {
      setState(() => _errorMessage = 'Passwords do not match.');
      return;
    }
    setState(() { _isLoading = true; _errorMessage = null; });

    final result = await ApiService.signUp(
      fullName: _nameController.text.trim(),
      email: _emailController.text.trim(),
      password: _passwordController.text,
    );

    if (!mounted) return;
    setState(() => _isLoading = false);

    if (result['status'] == 201) {
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (_) => const MainNavigation()),
      );
    } else {
      setState(() {
        _errorMessage = result['data']['error'] ?? 'Sign up failed.';
      });
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
              const SizedBox(height: 48),
              const Text('Create Account',
                  style: TextStyle(fontSize: 30, fontWeight: FontWeight.bold, color: Color(0xFF1A1A2E))),
              const SizedBox(height: 8),
              const Text('Join us and start communicating',
                  style: TextStyle(fontSize: 15, color: Colors.grey)),
              const SizedBox(height: 32),
              CustomTextField(label: 'Full Name', hintText: 'Enter your name', controller: _nameController),
              const SizedBox(height: 20),
              CustomTextField(label: 'Email', hintText: 'Enter your email',
                  controller: _emailController, keyboardType: TextInputType.emailAddress),
              const SizedBox(height: 20),
              CustomTextField(label: 'Password', hintText: 'Enter your password',
                  controller: _passwordController, obscureText: true),
              const SizedBox(height: 20),
              CustomTextField(label: 'Confirm Password', hintText: 'Confirm your password',
                  controller: _confirmPasswordController, obscureText: true),
              if (_errorMessage != null) ...[
                const SizedBox(height: 12),
                Text(_errorMessage!, style: const TextStyle(color: Colors.red, fontSize: 13)),
              ],
              const SizedBox(height: 32),
              CustomButton(text: 'Sign Up', onPressed: _signUp, isLoading: _isLoading),
              const SizedBox(height: 16),
              const Text(
                'By signing up, you agree to our Terms of\nService and Privacy Policy',
                style: TextStyle(fontSize: 12, color: Colors.grey, height: 1.5),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 20),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Text("Already have an account? ",
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
