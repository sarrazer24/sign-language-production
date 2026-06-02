import 'dart:typed_data';
import 'package:path_provider/path_provider.dart';

Future<Uint8List> fetchBlobBytes(String blobUrl) async {
  throw UnsupportedError('fetchBlobBytes is web-only');
}

Future<String> getTempDirPath() async {
  final dir = await getTemporaryDirectory();
  return dir.path;
}