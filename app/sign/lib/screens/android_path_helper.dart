import 'package:path_provider/path_provider.dart';

Future<String> getTempDirPath() async {
  final dir = await getTemporaryDirectory();
  return dir.path;
}