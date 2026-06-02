// ignore: avoid_web_libraries_in_flutter
import 'dart:html' as html;
import 'dart:typed_data';

Future<Uint8List> fetchBlobBytes(String blobUrl) async {
  final response = await html.HttpRequest.request(
    blobUrl,
    responseType: 'arraybuffer',
  );
  final buffer = response.response as dynamic;
  return Uint8List.view(buffer as ByteBuffer);
}

Future<String> getTempDirPath() async => '';