import 'package:flutter/material.dart';
import '../../widgets/custom_text_field.dart';
import '../../widgets/custom_button.dart';
import '../../services/api_service.dart';
import '../home/home_screen.dart';
import 'sign_up_screen.dart';
import 'forgot_password_screen.dart';
import '../../widgets/main_navigation.dart';

class SignInScreen extends StatefulWidget {
  const SignInScreen({super.key});

  @override
  State<SignInScreen> createState() => _SignInScreenState();
}

class _SignInScreenState extends State<SignInScreen> {
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();
  bool _rememberMe = false;
  bool _isLoading = false;
  String? _errorMessage;

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  void _signIn() async {
    setState(() { _isLoading = true; _errorMessage = null; });

    final result = await ApiService.signIn(
      email: _emailController.text.trim(),
      password: _passwordController.text,
    );

    if (!mounted) return;
    setState(() => _isLoading = false);

    if (result['status'] == 200) {
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (_) => const MainNavigation()),
      );
    } else {
      setState(() {
        _errorMessage = result['data']['error'] ?? 'Sign in failed.';
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
              const SizedBox(height: 60),
              const Text('Welcome back',
                  style: TextStyle(fontSize: 30, fontWeight: FontWeight.bold, color: Color(0xFF1A1A2E))),
              const SizedBox(height: 8),
              const Text('Sign in to continue',
                  style: TextStyle(fontSize: 15, color: Colors.grey)),
              const SizedBox(height: 40),
              CustomTextField(label: 'Email', hintText: 'Enter your email',
                  controller: _emailController, keyboardType: TextInputType.emailAddress),
              const SizedBox(height: 20),
              CustomTextField(label: 'Password', hintText: 'Enter your password',
                  controller: _passwordController, obscureText: true),
              const SizedBox(height: 14),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Row(children: [
                    Checkbox(value: _rememberMe,
                        onChanged: (val) => setState(() => _rememberMe = val ?? false),
                        activeColor: const Color(0xFF5B4FCF),
                        materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                        visualDensity: VisualDensity.compact),
                    const Text('Remember Me', style: TextStyle(fontSize: 13, color: Colors.grey)),
                  ]),
                  GestureDetector(
                    onTap: () => Navigator.push(context,
                        MaterialPageRoute(builder: (_) => const ForgotPasswordScreen())),
                    child: const Text('Forgot Password?',
                        style: TextStyle(fontSize: 13, color: Colors.grey)),
                  ),
                ],
              ),
              if (_errorMessage != null) ...[
                const SizedBox(height: 12),
                Text(_errorMessage!, style: const TextStyle(color: Colors.red, fontSize: 13)),
              ],
              const SizedBox(height: 36),
              CustomButton(text: 'Sign In', onPressed: _signIn, isLoading: _isLoading),
              const SizedBox(height: 24),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Text("Don't have an account? ",
                      style: TextStyle(fontSize: 14, color: Colors.grey)),
                  GestureDetector(
                    onTap: () => Navigator.push(context,
                        MaterialPageRoute(builder: (_) => const SignUpScreen())),
                    child: const Text('Sign Up',
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
